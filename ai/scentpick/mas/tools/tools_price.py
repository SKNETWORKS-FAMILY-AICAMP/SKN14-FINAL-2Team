# scentpick/mas/tools/tools_price.py
from langchain_core.tools import tool
import requests, re
from typing import Optional, List, Dict, Union

from ..tools.tools_keywords import extract_search_keyword_with_llm
from ..config import naver_client_id, naver_client_secret

def _remove_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "")

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _price_to_int(v) -> Optional[int]:
    try:
        s = re.sub(r"[^\d]", "", str(v))
        return int(s) if s else None
    except Exception:
        return None

def _title_match(title: str, brand: str, name: str) -> bool:
    """브랜드/제품명 토큰이 제목에 충분히 포함되는지(느슨 매칭)"""
    t = _norm(title)
    b = _norm(brand)
    n = _norm(name)
    tokens = [w for w in (b + " " + n).split() if len(w) >= 2]
    if not tokens:
        return True
    hit = sum(1 for w in tokens if w in t)
    return hit >= max(2, min(4, len(tokens) // 2))  # 절반 이상 매칭

@tool
def price_tool(
    user_query: str,
    brand: Optional[str] = None,
    name: Optional[str] = None,
    size_ml: Optional[int] = None,
    budget_krw: Optional[int] = None,
    topk_fetch: int = 10,    # API에서 가져올 개수
    topk_return: int = 1,    # 실제 반환 개수(최저가 우선)
    return_json: bool = False,  # JSON 모드
    name_regex: Optional[str] = None,  # 제목 필터용 정규식(예: r"\bno\s*5\b")
) -> Union[str, Dict]:
    """
    네이버 쇼핑 API로 향수 가격 조회.
    - brand/name/size_ml 있으면 정밀 검색(제목 느슨매칭)
    - name_regex가 주어지면 제목 정규식 필터를 우선 적용
    - return_json=True: {"query":..., "items":[{"title","price"}], "under_budget":[...]} 반환
    """

    # 1) 질의어 구성
    if brand and name:
        q = f"{brand} {name}".strip()
        if size_ml and size_ml > 0:
            q = f"{q} {int(size_ml)}ml"
        search_keyword = q
    else:
        # LLM이 만든 짧고 잘 먹히는 키워드 사용
        search_keyword = extract_search_keyword_with_llm(user_query)

    # 2) API 호출
    url = "https://openapi.naver.com/v1/search/shop.json"
    headers = {
        "X-Naver-Client-Id": naver_client_id,
        "X-Naver-Client-Secret": naver_client_secret,
    }
    params = {
        "query": search_keyword,
        "display": min(max(int(topk_fetch), 1), 30),
        "sort": "sim",
    }

    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
    except Exception as e:
        return {"error": f"request_error: {e}"} if return_json else f"❌ 요청 오류: {e}"

    data = r.json() or {}
    raw_items: List[Dict] = data.get("items") or []
    if not raw_items:
        if return_json:
            return {"query": search_keyword, "items": [], "under_budget": []}
        return f"😔 '{search_keyword}'에 대한 검색 결과가 없습니다.\n💡 다른 브랜드명이나 향수명으로 다시 검색해보세요."

    # 3) 가공: 제목/가격 + (옵션) 제목 정규식 + (옵션) 브랜드/제품 느슨매칭
    view: List[Dict] = []
    patt = re.compile(name_regex, flags=re.IGNORECASE) if name_regex else None

    for it in raw_items:
        title = _remove_html_tags(it.get("title", ""))
        price = _price_to_int(it.get("lprice"))
        if not title or not price:
            continue
        if patt and not patt.search(title):
            continue
        if brand and name and not _title_match(title, brand, name):
            continue
        view.append({"title": title, "price": price})

    if not view:
        if return_json:
            return {"query": search_keyword, "items": [], "under_budget": []}
        out = f"🔍 '{search_keyword}' 검색 결과:\n\n조건(제목 매칭/필터)에 부합하는 항목이 없습니다."
        return out

    # 4) 최저가 우선 정렬 후 상위 N개
    view.sort(key=lambda x: x["price"])
    top = view[:max(1, int(topk_return))]

    if return_json:
        under = [x for x in top if (budget_krw is None or x["price"] <= int(budget_krw))]
        return {"query": search_keyword, "items": top, "under_budget": under}

    # 5) 텍스트 모드(호환)
    output = f"🔍 '{search_keyword}' 검색 결과:\n\n"
    prices = []
    for i, p in enumerate(top, 1):
        output += f"{i}. {p['title']}\n   ₩ {p['price']:,}원\n\n"
        prices.append(p["price"])

    if len(prices) >= 2:
        output += "**가격대 정보**\n"
        output += f"   💰 검색된 가격 범위: {min(prices):,}원 ~ {max(prices):,}원\n"
    else:
        output += "**참고사항**\n"

    output += "   ⚠️ 가격은 변동될 수 있으니 각 쇼핑몰에서 직접 확인하세요.\n"
    if budget_krw is not None:
        output = f"(예산 ≤ {int(budget_krw):,}원)\n\n" + output
    return output
