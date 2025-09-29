import json
import logging
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
import openai
import os
from datetime import datetime, timezone

from ..state import AgentState
from ..config import llm
from ..tools.price_parse import extract_budget_krw
from ..tools.tools_price import price_tool  # LangChain Tool(.invoke)

logger = logging.getLogger(__name__)

# API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# 파인콘 초기화
pc = Pinecone(api_key=pinecone_api_key)
REVIEW_INDEX_NAME = "review-vectordb"
# 🔁 벡터DB2 사용
PERFUME_INDEX_NAME = "perfume-vectordb2"
review_index = pc.Index(REVIEW_INDEX_NAME)
perfume_index = pc.Index(PERFUME_INDEX_NAME)

# 🔽 유사도 임계값 (None이면 필터 미적용)
MIN_SIMILARITY_THRESHOLD = None

# ---------------------------
# 헬퍼
# ---------------------------

def _get_meta(meta: dict, *keys, default: Optional[str] = "") -> str:
    """여러 키 후보 중 먼저 매칭되는 값을 반환."""
    if not meta:
        return default or ""
    for k in keys:
        val = meta.get(k)
        if val is not None:
            return str(val)
    return default or ""

def _get_from_dotpath(d: dict, dotkey: str) -> Optional[Any]:
    """
    중첩 dict에서 'a.b.c' 형태 경로로 안전하게 값 추출.
    존재하지 않으면 None.
    """
    try:
        cur = d
        for part in dotkey.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur
    except Exception:
        return None

def _get_from_paths(meta: dict, paths: List[str]) -> Optional[Any]:
    """여러 후보 도트경로 중 먼저 성공하는 값을 반환."""
    if not meta:
        return None
    for p in paths:
        v = _get_from_dotpath(meta, p) if "." in p else meta.get(p)
        if v is not None and (not isinstance(v, str) or v.strip() != ""):
            return v
    return None

def _to_int_str(v: Any) -> Optional[str]:
    """376, '376', 376.0 -> '376' 로 정규화. 실패 시 None."""
    if v is None:
        return None
    try:
        return str(int(float(v)))
    except Exception:
        return None

def _get_no_as_intstr(meta: dict) -> Optional[str]:
    """
    vectordb2 메타에서 내부 링크용 id로 쓸 'no'를 꺼내 정수문자열로 변환.
    우선순위: p.no -> perfume_data.no -> no
    """
    candidate = _get_from_paths(meta, ["p.no", "perfume_data.no", "no"])
    return _to_int_str(candidate)

def _get_any_id(meta: dict, *keys) -> Optional[str]:
    """
    Pinecone metadata에서 id 유사 키를 문자열로 뽑아냄 (fallback).
    예: ("id", "perfume_id", "pid")
    """
    if not meta:
        return None
    for k in keys:
        if k in meta and meta[k] is not None:
            s = str(meta[k]).strip()
            if s != "":
                return s
    return None

def get_openai_embedding(text: str) -> List[float]:
    """OpenAI 임베딩 모델로 텍스트 벡터화"""
    try:
        resp = openai.embeddings.create(model="text-embedding-ada-002", input=text)
        return resp.data[0].embedding
    except Exception as e:
        logger.error(f"[get_openai_embedding] Error: {e}", exc_info=True)
        raise

# ---------------------------
# 파이프라인 함수
# ---------------------------
def parse_user_query(user_query: str) -> Dict[str, Any]:
    """사용자 쿼리를 향 설명과 가격대로 분리"""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "사용자의 향수 질문을 JSON으로 변환하세요. "
         "향 설명은 scent_description, 가격 관련 내용은 price_query에 넣으세요. "
         "형태: {{\"scent_description\": \"...\", \"price_query\": \"...\"}}"),
        ("user", "{query}")
    ])
    try:
        chain = prompt | llm
        response = chain.invoke({"query": user_query})
        parsed = json.loads(getattr(response, "content", "{}"))
        return {
            "scent_description": parsed.get("scent_description", user_query),
            "price_query": parsed.get("price_query", "")
        }
    except Exception as e:
        logger.error(f"[parse_user_query] Error: {e}", exc_info=True)
        return {"scent_description": user_query, "price_query": ""}

def search_review_vectordb(scent_description: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """review-vectordb에서 향 설명 기반 RAG 검색 (메타키 방어 포함)"""
    try:
        logger.info(f"[search_review_vectordb] Searching for: {scent_description}")
        query_embedding = get_openai_embedding(scent_description)
        results = review_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        matches = results.get("matches") or []
        if not matches:
            logger.warning("[search_review_vectordb] No matches found")
            return []

        rag_results = []
        for i, match in enumerate(matches):
            meta = (match.get("metadata") or {})
            content = _get_meta(meta, "content", "text", "review", "body")
            brand   = _get_meta(meta, "brand", "Brand")
            name    = _get_meta(meta, "name", "Name", "title")
            score   = match.get("score", 0.0)

            if not (content or brand or name):
                logger.warning(f"[search_review_vectordb] Skip empty meta: {meta}")
                continue

            rag_results.append({
                "content": content,
                "brand": brand,
                "name": name,
                "score": score,
                "metadata": meta
            })

        logger.info(f"[search_review_vectordb] Collected {len(rag_results)} usable results")
        return rag_results
    except Exception as e:
        logger.error(f"[search_review_vectordb] Error: {e}", exc_info=True)
        return []

def analyze_rag_results(scent_description: str, rag_results: List[Dict]) -> Dict[str, Any]:
    usable = [r for r in rag_results if any([r.get("content"), r.get("brand"), r.get("name")])]
    if not usable:
        return {"analyzed_scent": scent_description, "confidence": 0.0}

    rag_text = ""
    for i, r in enumerate(usable, 1):
        b = r.get("brand", "")
        n = r.get("name", "")
        c = (r.get("content") or "").strip()
        if len(c) > 600:
            c = c[:600] + " ..."
        rag_text += f"{i}. {b} {n}: {c}\n"

    # 🔧 예시 JSON 중괄호 이스케이프 필수
    prompt = ChatPromptTemplate.from_messages([
        ("system", """다음은 향수 리뷰 데이터베이스에서 검색된 결과입니다.
사용자가 원하는 향의 특성을 분석해서 JSON으로 반환해주세요.

형태:
{{
    "analyzed_scent": "분석된 향의 특성 (구체적이고 상세하게)",
    "confidence": 0.0~1.0
}}

반드시 JSON 형태로만 응답하세요."""),
        ("user", "사용자 요청: {scent_description}\n\n검색 결과:\n{rag_text}")
    ])

    try:
        chain = prompt | llm
        response = chain.invoke({
            "scent_description": scent_description,
            "rag_text": rag_text
        })
        parsed = json.loads(getattr(response, "content", "{}"))
        return {
            "analyzed_scent": parsed.get("analyzed_scent", scent_description),
            "confidence": float(parsed.get("confidence", 0.5))
        }
    except Exception as e:
        logger.error(f"[analyze_rag_results] Error: {e}", exc_info=True)
        return {"analyzed_scent": scent_description, "confidence": 0.0}

def search_perfume_vectordb(analyzed_scent: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """perfume-vectordb2에서 분석된 향 특성으로 유사도 검색"""
    try:
        query_embedding = get_openai_embedding(analyzed_scent)
        results = perfume_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        matches = results.get("matches") or []
        if not matches:
            logger.warning("[search_perfume_vectordb] No matches at all")
            return []

        # 디버깅 로그
        for i, m in enumerate(matches[:5], 1):
            logger.info(f"[perfume_vdb] top{i} score={m.get('score')} meta_keys={list((m.get('metadata') or {}).keys())[:5]}")

        perfumes = []
        for match in matches[:5]:     # 상위 5개 채택
            meta = match.get("metadata", {}) or {}
            score = match.get("score", 0.0)

            if MIN_SIMILARITY_THRESHOLD is not None and score is not None:
                if score < MIN_SIMILARITY_THRESHOLD:
                    continue

            # ✅ vectordb2: p.no → id 로 사용 (없으면 perfume_data.no → no → 기타 키)
            id_from_no = _get_no_as_intstr(meta)
            pid = id_from_no or _get_any_id(meta, "id", "perfume_id", "pid")

            perfumes.append({
                "id": pid,  # 내부 링크용
                "brand": _get_meta(meta, "brand", "Brand"),
                "name":  _get_meta(meta, "name", "Name", "title"),
                "score": float(score or 0.0),
                "size_ml": meta.get("size_ml"),
                "metadata": meta
            })

        # 필터로 다 날렸다면 top3는 무조건 살림
        if not perfumes:
            for match in matches[:3]:
                meta = match.get("metadata", {}) or {}
                pid = _get_no_as_intstr(meta) or _get_any_id(meta, "id", "perfume_id", "pid")
                perfumes.append({
                    "id": pid,
                    "brand": _get_meta(meta, "brand", "Brand"),
                    "name":  _get_meta(meta, "name", "Name", "title"),
                    "score": float(match.get("score") or 0.0),
                    "size_ml": meta.get("size_ml"),
                    "metadata": meta
                })
            logger.info(f"[perfume_vdb] Fallback: kept top{len(perfumes)} without threshold")

        return perfumes
    except Exception as e:
        logger.error(f"[search_perfume_vectordb] Error: {e}", exc_info=True)
        return []

def check_prices_and_filter(perfume_list: List[Dict], budget_info: Dict) -> List[Dict]:
    """향수 리스트의 가격을 조회하고 예산 내 필터링 (price_tool.invoke 사용)"""
    if not perfume_list:
        return []
    
    logger.info(f"[check_prices_and_filter] Processing {len(perfume_list)} perfumes")
    logger.info(f"[check_prices_and_filter] Budget info: {budget_info}")
    
    budget_matched = []
    
    for perfume in perfume_list:
        brand = (perfume.get('brand') or "").strip()
        name  = (perfume.get('name')  or "").strip()
        size_ml = perfume.get('size_ml')

        if not (brand and name):
            logger.warning(f"[check_prices_and_filter] Skip: missing brand/name: {perfume}")
            continue

        try:
            size_part = f" {int(size_ml)}ml" if size_ml else ""
            query_str = f"{brand} {name}{size_part}".strip()

            # LangChain Tool 호출
            tool_res = price_tool.invoke({
                "user_query": query_str,
                "brand": brand,
                "name": name,
                "size_ml": size_ml,
                "topk_fetch": 10,
                "topk_return": 3,
                "return_json": True
            })

            logger.info(f"[check_prices_and_filter] Price result: {tool_res}")

            if isinstance(tool_res, dict) and tool_res.get("items"):
                cheapest_item = tool_res["items"][0]
                price = cheapest_item.get("price")
                title = cheapest_item.get("title")
                link  = cheapest_item.get("link") or cheapest_item.get("url")

                # ✅ id 그대로 보존 (price_tool은 id를 주지 않으므로 우리가 받은 걸 유지)
                perfume["id"] = perfume.get("id")
                perfume["price"] = price
                perfume["price_title"] = title
                perfume["detail_url"] = link

                # 예산 체크
                is_within_budget = True
                if budget_info:
                    is_within_budget = False
                    if budget_info.get("budget"):
                        budget = budget_info["budget"]
                        op = budget_info.get("budget_op", "eq")
                        if op == "lte" and price <= budget:
                            is_within_budget = True
                        elif op == "gte" and price >= budget:
                            is_within_budget = True
                        elif op == "eq" and abs(price - budget) <= budget * 0.2:  # ±20% 허용
                            is_within_budget = True
                    elif budget_info.get("budget_min") and budget_info.get("budget_max"):
                        if budget_info["budget_min"] <= price <= budget_info["budget_max"]:
                            is_within_budget = True

                if is_within_budget:
                    budget_matched.append(perfume)

            else:
                logger.warning(f"[check_prices_and_filter] No price items for {brand} {name}")
                if not budget_info:
                    perfume["id"] = perfume.get("id")
                    perfume["price"] = None
                    perfume["price_title"] = "가격 정보 없음"
                    budget_matched.append(perfume)

        except Exception as e:
            logger.exception(f"[check_prices_and_filter] price_tool failed for {brand} {name}: {e}")
            if not budget_info:
                perfume["id"] = perfume.get("id")
                perfume["price"] = None
                perfume["price_title"] = "가격 조회 실패"
                budget_matched.append(perfume)
            continue
    
    logger.info(f"[check_prices_and_filter] Final matched: {len(budget_matched)}/{len(perfume_list)}")
    return budget_matched

def generate_perfume_response(budget_matched: List[Dict], budget_info: Dict, scent_description: str):
    """
    LLM_parser와 동일 톤으로 본문 생성 + rec_echo 호환 items 반환
    return: (response_text, rec_items)
    """
    header = f"🌸 '{scent_description}' 취향에 맞는 향수를 찾았어요!\n\n"

    # 예산 라벨
    budget_label = ""
    if budget_info:
        if budget_info.get("budget"):
            budget = budget_info["budget"]
            op = budget_info.get("budget_op", "eq")
            if op == "lte":
                budget_label = f"💰 예산 {budget:,}원 이하 범위 내 추천:\n\n"
            elif op == "gte":
                budget_label = f"💰 예산 {budget:,}원 이상 범위 내 추천:\n\n"
            else:
                budget_label = f"💰 예산 약 {budget:,}원 범위 내 추천:\n\n"
        elif budget_info.get("budget_min") and budget_info.get("budget_max"):
            budget_label = f"💰 예산 {budget_info['budget_min']:,}원~{budget_info['budget_max']:,}원 범위 내 추천:\n\n"

    # 가격순 정렬
    sorted_perfumes = sorted(budget_matched, key=lambda x: x.get("price", float("inf")))

    lines = []
    rec_items = []
    for i, p in enumerate(sorted_perfumes, 1):
        brand = (p.get("brand") or "").strip()
        name  = (p.get("name")  or "").strip()
        price = p.get("price")
        price_title = p.get("price_title") or ""
        score = p.get("score", 0.0)

        lines.append(f"{i}. **{brand} {name}**")
        if price is not None:
            lines.append(f"   💰 최저가: {price:,}원")
            if price_title:
                display_title = price_title[:50] + "..." if len(price_title) > 50 else price_title
                lines.append(f"   🛒 상품: {display_title}")
        else:
            lines.append(f"   💰 가격: {price_title or '가격 정보 없음'}")
        lines.append(f"   🎯 유사도: {score:.2f}\n")

        rec_items.append({
            "id": p.get("id"),                 # 내부링크용 id (p.no 기반 정수 문자열)
            "brand": brand,
            "name": name,
            "detail_url": p.get("detail_url"), # 외부 링크(있을 경우)
        })

    footer = (
        "📝 **참고사항**\n"
        "• 가격은 변동될 수 있으니 구매 전 확인하세요\n"
        "• 향수는 개인차가 있으니 샘플 테스트를 권장합니다"
    )
    body = header + budget_label + "\n".join(lines) + footer
    return body, rec_items

def generate_final_llm_response(user_query: str, scent_description: str, price_query: str) -> str:
    """조건에 맞는 향수가 없을 때 LLM이 직접 추천"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """향수 전문가로서 사용자의 요청에 맞는 향수를 추천해주세요.
구체적인 브랜드명과 제품명, 가격대, 향의 특징을 포함해서 3개 정도 추천해주세요.
친근하고 전문적인 톤으로 답변해주세요."""), 
        ("user", "전체 요청: {user_query}\n향 취향: {scent_description}\n가격대: {price_query}")
    ])
    try:
        chain = prompt | llm
        response = chain.invoke({
            "user_query": user_query,
            "scent_description": scent_description,
            "price_query": price_query
        })
        llm_content = getattr(response, "content", "")
        final_response = llm_content + "\n\n" + \
            "⚠️ **안내사항**\n" \
            "이 추천은 AI 추천으로 저희 scentpick에 없는 데이터일 수도 있습니다.\n" \
            "자세한 상품은 직접 쇼핑몰에서 확인하시기 바랍니다."
        return final_response
    except Exception as e:
        logger.error(f"[generate_final_llm_response] Error: {e}", exc_info=True)
        return "죄송합니다. 추천을 생성하는 중에 오류가 발생했습니다."

def is_review_agent_query(query: str) -> bool:
    # 라우터에서 scent+price 조합만 review로 보내도록 제어
    return True

def review_agent_node(state: AgentState) -> AgentState:
    try:
        # 최신 사용자 메시지
        messages = state.get("messages", [])
        user_query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content or ""
                break
        if not user_query:
            ans = "질문을 다시 입력해주세요."
            return {
                "messages": [AIMessage(content=ans)],
                "final_answer": ans,
                "perfume_list": [],
                "search_results": {"matches": []},
                "rec_history": [],
                "last_agent": "review_agent"
            }

        # 1) 파싱
        parsed_query = parse_user_query(user_query)
        scent_description = parsed_query["scent_description"]
        price_query = parsed_query["price_query"]
        budget_info = extract_budget_krw(price_query) if price_query else {}

        # 2) 리뷰 RAG
        rag_results = search_review_vectordb(scent_description)
        if not rag_results:
            llm_response = generate_final_llm_response(user_query, scent_description, price_query)
            summary = f"\n💬 추천 결과:\n\n{llm_response}"
            return {
                "messages": [AIMessage(content=summary)],
                "final_answer": summary,
                "perfume_list": [],
                "search_results": {"matches": []},
                "rec_history": [],
                "last_agent": "review_agent"
            }

        # 3) 분석
        analysis_result = analyze_rag_results(scent_description, rag_results)
        analyzed_scent = analysis_result["analyzed_scent"]

        # 4) 향수 후보 (id는 p.no 기반)
        perfume_candidates = search_perfume_vectordb(analyzed_scent)
        if not perfume_candidates:
            llm_response = generate_final_llm_response(user_query, scent_description, price_query)
            summary = f"\n💬 추천 결과:\n\n{llm_response}"
            return {
                "messages": [AIMessage(content=summary)],
                "final_answer": summary,
                "perfume_list": [],
                "search_results": {"matches": []},
                "rec_history": [],
                "last_agent": "review_agent"
            }

        # 5) 리스트 정리 (id 포함)
        perfume_list = [{
            "id": p.get("id"),            # p.no → 정수 문자열
            "brand": p.get("brand", ""),
            "name": p.get("name", ""),
            "score": p.get("score", 0.0),
            "size_ml": p.get("size_ml"),
        } for p in perfume_candidates]

        # 6) 가격 조회 + 예산 필터 (id 유지)
        budget_matched = check_prices_and_filter(perfume_list, budget_info)

        # 7) 응답 생성
        if budget_matched:
            body, rec_items = generate_perfume_response(budget_matched, budget_info, scent_description)
            summary = f"\n💬 추천 결과:\n\n{body}"

            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "source": "review_agent",
                "items": rec_items,  # rec_echo 사용 (id 포함)
            }

            return {
                "messages": [AIMessage(content=summary)],
                "final_answer": summary,
                "perfume_list": budget_matched,     # 프론트 노출 (id 포함)
                "search_results": {"matches": []},  # 중복 복구 방지
                "rec_history": [entry],             # rec_echo 호환
                "last_agent": "review_agent",
            }
        else:
            # 조건 미매칭 → 안내 + LLM 백업
            if budget_info:
                if budget_info.get("budget"):
                    budget = budget_info["budget"]
                    op = budget_info.get("budget_op", "eq")
                    budget_text = f"{budget:,}원 {'이하' if op == 'lte' else '이상' if op == 'gte' else '대'}"
                else:
                    budget_text = f"{budget_info.get('budget_min', 0):,}원~{budget_info.get('budget_max', 0):,}원"
                fallback_msg = (
                    f"😔 '{scent_description}' 취향과 {budget_text} 예산에 맞는 향수를 찾지 못했어요.\n\n"
                    "💡 다음을 시도해보세요:\n"
                    "• 예산 범위를 조금 늘려보세요\n"
                    "• 향의 특성을 다르게 표현해보세요\n"
                    "• 구체적인 브랜드나 제품명을 알려주세요\n\n"
                )
            else:
                fallback_msg = "😔 조건에 맞는 향수를 찾지 못했어요.\n\n"

            llm_response = generate_final_llm_response(user_query, scent_description, price_query)
            summary = f"\n💬 추천 결과:\n\n{fallback_msg}🤖 **AI 추천**:\n{llm_response}"

            return {
                "messages": [AIMessage(content=summary)],
                "final_answer": summary,
                "perfume_list": [],                  # 리스트 비움
                "search_results": {"matches": []},   # 복구 방지
                "rec_history": [],
                "last_agent": "review_agent",
            }

    except Exception as e:
        logger.error(f"[review_agent_node] Error: {e}", exc_info=True)
        ans = "죄송합니다. 추천 과정에서 오류가 발생했습니다. 다시 시도해주세요."
        return {
            "messages": [AIMessage(content=ans)],
            "final_answer": ans,
            "perfume_list": [],
            "search_results": {"matches": []},
            "rec_history": [],
            "last_agent": "review_agent",
            "last_error": str(e)
        }
