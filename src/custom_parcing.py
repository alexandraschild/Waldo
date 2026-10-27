import re

def normalize_answer(ans: str) -> str:
    if not ans:
        return ""
        
    ans = ans.strip().lower()
    ans = re.sub(r"^(answer\s*[:\-]\s*)", "", ans)
    ans = re.sub(r"[^\w\s]", "", ans)
    
    return ans


def mmeval_parse_answer(pred_ans: str):
    ans = normalize_answer(pred_ans)

    if ans.startswith("yes"):
        return "yes"
    if ans.startswith("no"):
        return "no"

    return ans


def mathverse_parse_answer(pred_ans: str):
    ans = normalize_answer(pred_ans)

    match = re.match(r"[abcd]", ans)
    if match:
        return match.group(0)

    return None