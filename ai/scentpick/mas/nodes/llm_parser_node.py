from langchain_core.messages import HumanMessage, AIMessage
from ..state import AgentState
from ..tools.tools_parsers import run_llm_parser
from ..tools.tools_metafilters import apply_meta_filters
from ..tools.tools_rag import query_pinecone, generate_response
import json
from ..config import llm, embeddings
from ..tools.tools_price import price_tool
from ..tools.vector_db_utils import build_item_queries_from_vectordb
from datetime import datetime, timezone

def _to_int_ml(v):
    try:
        if v is None: return None
        if isinstance(v, (int, float)): return int(v)
        s = str(v).lower().replace("ml", "").strip()
        return int(float(s))
    except Exception:
        return None

def _extract_candidates(search_results: dict, preferred_size=None, top_n=5):
    matches = (search_results or {}).get("matches", []) or []
    items = []
    for m in matches[: top_n*2]:  # 여유로 넉넉히 가져와서 size 필터
        meta = (m or {}).get("metadata", {}) or {}
        brand = meta.get("brand") or meta.get("Brand") or ""
        name  = meta.get("name")  or meta.get("Name")  or ""
        url   = meta.get("detail_url") or meta.get("url") or meta.get("detailUrl") or None
        size  = _to_int_ml(meta.get("size") or meta.get("size_ml") or meta.get("Size"))
        cand = {"brand": brand, "name": name, "size": size, "detail_url": url}
        items.append(cand)

    if preferred_size is not None:
        ps = _to_int_ml(preferred_size)
        filtered = [it for it in items if it.get("size") == ps]
        if filtered:
            items = filtered

    # 비어있으면 name 없는 항목은 제거
    items = [it for it in items if it.get("name")]
    return items[:top_n]

def LLM_parser_node(state: AgentState) -> AgentState:
    """RAG 파이프라인 + (있다면) 가격 검색, 그리고 rec_history 누적"""
    # 0) 최신 사용자 메시지
    user_query = "(empty)"
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            user_query = m.content
            break

    try:
        print(f"🔍 LLM_parser 실행: {user_query}")

        # 1) LLM 파싱
        print(f"🔧 run_llm_parser 호출")
        parsed_json = run_llm_parser(user_query)
        print(f"🔧 run_llm_parser 결과")
        print(f"{json.dumps(parsed_json, ensure_ascii=False)}")

        # 2) 메타필터
        print(f"🔧 apply_meta_filters 호출")
        filtered_json = apply_meta_filters(parsed_json)
        print(f"🔧 apply_meta_filters 결과")
        print(f"{json.dumps(filtered_json, ensure_ascii=False)}")

        # 3) 쿼리 벡터화
        print(f"쿼리 벡터화")
        query_vector = embeddings.embed_query(user_query)
        

        # 4) Pinecone 검색
        print(f"pinecone 검색")
        n_recs = int(parsed_json.get("recommendation_count") or 3)  # 기본값 3
        search_results = query_pinecone(query_vector, filtered_json, top_k=n_recs)
        if hasattr(search_results, "to_dict"):
            search_results = search_results.to_dict()
        print("Pinecone 검색 결과 (메타필터링 컬럼만)")
        matches = (search_results or {}).get("matches", [])
        if not matches:
            print("결과 없음")
        else:
            for i, m in enumerate(matches, 1):
                meta = m.get("metadata", {}) or {}
                brand = meta.get("brand", "정보없음")
                name = meta.get("name", "정보없음")
                gender = meta.get("gender", "정보없음")
                sizes = meta.get("sizes", "정보없음")
                season = meta.get("season_score", "정보없음")
                day_night = meta.get("day_night_score", "정보없음")
                concentration = meta.get("concentration", "정보없음")
                
                print(f"{i}. brand={brand}, name={name}, gender={gender}, size={sizes}ml, "
                    f"season={season}, day_night={day_night}, concentration={concentration}")


        # 4-1) 추천 후보 추출 (rec_echo용 표준 스키마)
        preferred_size = parsed_json.get("sizes")
        candidates = _extract_candidates(search_results, preferred_size=preferred_size, top_n=n_recs)
        print("추천 후보 (정제된 candidates):")
        if not candidates:
            print("결과 없음")
        else:
            for i, it in enumerate(candidates, 1):
                brand = it.get("brand", "정보없음")
                name  = it.get("name", "정보없음")
                size  = it.get("size", "정보없음")
                url   = it.get("detail_url") or ""
                print(f"{i}. {brand} - {name} ({size}ml) {url}")

        # 최종 응답용 문자열
        final_response_lines = []
        for i, it in enumerate(candidates, 1):
            brand = it.get("brand", "정보없음")
            name = it.get("name", "정보없음")
            size = it.get("size", "정보없음")
            line = f"{i}. {brand} - {name} ({size}ml)"
            final_response_lines.append(line)


        # 5) 최종 사용자 답변 텍스트 생성
        final_response = generate_response(user_query, search_results, limit=n_recs)

        # 6) 가격 의도 감지
        price_keywords_ko = ['가격', '얼마', '가격대', '구매', '판매', '할인', '어디서 사', '어디서사', '배송비', '최저가']
        price_keywords_en = ['price', 'cost', 'cheapest', 'buy', 'purchase', 'discount']
        lower = user_query.lower()
        has_price_intent = any(k in user_query for k in price_keywords_ko) or any(k in lower for k in price_keywords_en)

        if has_price_intent and candidates:
            # 벡터DB에서 뽑힌 후보로만 가격 검색
            item_query_bundles = build_item_queries_from_vectordb(
                search_results=search_results,
                facets=parsed_json,
                top_n_items=min(n_recs, len(candidates))
            )
            price_sections = []
            for bundle in item_query_bundles:
                label = bundle["item_label"]  # 예: "YSL Libre EDP 50ml"
                for q in bundle["queries"]:
                    try:
                        res = price_tool.invoke({"user_query": q})
                        if res:
                            price_sections.append(f"**{label}**\n{res}")
                            break
                    except Exception as e:
                        print(f"❌ 가격 검색 오류({q}): {e}")

            if price_sections:
                final_response_with_price = f"""{final_response}

---

💰 **가격 정보**\n
{'\n\n'.join(price_sections)}"""
            else:
                final_response_with_price = f"""{final_response}

---

💰 **가격 정보**\n
🔍 벡터DB에서 추천된 제품명으로 검색했지만, 일치 결과를 찾지 못했어요.
원하시는 **제품명 + 농도 + 용량(예: 50ml)** 조합으로 다시 알려주세요."""
        else:
            final_response_with_price = final_response

        # 7) 로그/요약(개발용): 사용자에게 그대로 보여주되, 한 메시지(델타)만 추가
#         summary = f"""[LLM_parser] RAG 파이프라인 완료 ✅

# 📊 파싱 결과: {json.dumps(parsed_json, ensure_ascii=False)}
# 🔍 필터링 결과: {json.dumps(filtered_json, ensure_ascii=False)}
# 🎯 검색된 향수 개수: {len((search_results or {}).get('matches', []))}

# 💬 추천 결과:
# {final_response_with_price}"""
        summary = f"""
💬 추천 결과:\n\n
{final_response_with_price}"""

        # 8) rec_history 누적 엔트리
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": "LLM_parser",
            "items": candidates,  # ← rec_echo가 그대로 읽음
        }

        return {
            "messages": [AIMessage(content=summary)],   # ✅ 이번 턴 델타만
            "parsed_slots": parsed_json,
            "search_results": search_results,
            "final_answer": final_response_with_price,
            "rec_history": [entry],                    # ✅ 리스트 누적
            "last_agent": "LLM_parser",
        }

    except Exception as e:
        error_msg = f"[LLM_parser] RAG 파이프라인 실행 중 오류: {e}"
        print(f"❌ LLM_parser 전체 오류: {e}")
        return {
            "messages": [AIMessage(content=error_msg)],
            "parsed_slots": {},
            "search_results": {"matches": []},
            "final_answer": error_msg,
            "last_agent": "LLM_parser",
        }
