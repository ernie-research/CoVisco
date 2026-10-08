import re

from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter

REPLACE_PROMPT = "Please answer directly with only the letter of the correct option and nothing else."


def realworldqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def realworldqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"]:
        question = question.replace(REPLACE_PROMPT, "")
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


# number_words_to_digits = {
#     "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
#     "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
#     "ten": "10"
# }


def realworldqa_process_results(doc, results):
    pred = results[0].lower().strip().rstrip(".")
    gt_ans = doc["answer"].lower().strip()

    print(f"Prediction: {pred}, Ground Truth: {gt_ans}")
    # assert gt_ans in ["a", "b", "c", "d"]
    score = 1.0 if pred == gt_ans else 0.0
    return {
        "exact_match": score,
    }


class NumberWordsToDigitsFilter(MapFilter):
    def __init__(self) -> None:
        mapping_dict = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
        super().__init__(mapping_dict, default_value=None)

    def apply(self, resps, docs):
        def filter_set(inst):
            return [self.mapping_dict.get(resp.lower(), resp) for resp in inst]

        return [filter_set(resp) for resp in resps]


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            # 解析题干选项，兼容 "A. 文字" 与 "A) 文字" 两种排版。
            option_regex = re.compile(r"(?<![A-Za-z0-9])([A-Z])[\.\)]\s+([^\n]*)")
            text_to_alpha = {}
            valid_letters = set()
            for letter, choice_text in option_regex.findall(doc["question"]):
                letter = letter.upper()
                valid_letters.add(letter)
                choice_clean = self._normalize(choice_text)
                if choice_clean:
                    text_to_alpha[choice_clean] = letter
            # 兜底：即使某个选项后面没有正文，也把字母收进合法集合。
            valid_letters |= {m.upper() for m in re.findall(r"(?<![A-Za-z0-9])([A-Z])[\.\)]", doc["question"])}

            filtered = [self._extract_answer(resp, valid_letters, text_to_alpha) for resp in r]
            filtered_resps.append(filtered[0])

        return filtered_resps

    @staticmethod
    def _normalize(text):
        """小写化、去标点、压缩空白，让选项正文与回答做对称比较。"""
        text = re.sub(r"[^\w\s]", "", text.lower())
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _extract_answer(cls, resp, valid_letters, text_to_alpha):
        """按置信度从高到低抽取选项字母，全部失败时回退为归一化后的回答。

        1) 整条回答就是一个选项字母（可带括号/尾标点）："B" / "B." / "(B)" / "B)"
        2) 以「字母+标签标点」开头："B. 文字" / "(C) 文字"（要求字母后紧跟 . ) : - ，
           因此不会把冠词开头的 "A red car" 误判成选项 A）
        3) 归一化后与某个选项正文完全相等
        4) 某个选项正文作为独立词组出现在回答中（如 "B Robbery" 命中正文 "Robbery"）
        5) 回答中出现且仅出现一个「带标点」的合法选项字母（如 "The answer is B."）
        6) 兜底：返回归一化后的回答
        """
        stripped = resp.strip()

        # 1) 独立字母
        m = re.match(r"^\(?\s*([A-Za-z])\s*[\.\):\-]*\s*$", stripped)
        if m and m.group(1).upper() in valid_letters:
            return m.group(1).upper()

        # 2) 以「字母+标签标点」开头
        m = re.match(r"^\(?\s*([A-Za-z])\s*[\.\):\-]\s+\S", stripped)
        if m and m.group(1).upper() in valid_letters:
            return m.group(1).upper()

        norm_resp = cls._normalize(resp)

        # 3) 与选项正文完全相等
        if norm_resp in text_to_alpha:
            return text_to_alpha[norm_resp]

        # 4) 选项正文作为独立词组命中
        for choice_clean, letter in text_to_alpha.items():
            if choice_clean and re.search(rf"(?<!\w){re.escape(choice_clean)}(?!\w)", norm_resp):
                return letter

        # 5) 回答中唯一且带标点的合法选项字母
        labeled = {mm.upper() for mm in re.findall(r"(?<![A-Za-z0-9])([A-Za-z])(?=[\.\):\-,]|$)", stripped)}
        labeled &= valid_letters
        if len(labeled) == 1:
            return next(iter(labeled))

        # 6) 兜底
        return norm_resp
