import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

from utils import *

BASE_MODEL_NAME = "cyberagent/open-calm-3b"
peft_model_id = "open-calm-3b-ft"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

instruction = "半年で15単位しか取れませんでした。除籍されますか？"
context = """(除籍)
第42条　次の各号の一に該当する者は、当該学部教授会の議を経て、学長が、これを除籍する。
(1)　死亡した者又は長期間にわたり行方不明の者
(2)　第8条に規定する在学期間を超えた者
[第8条]
(3)　第39条第4項及び第5項に規定する休学期間を超えて、なお復学できない者
[第39条第4項] [第5項]
(4)　病気その他の理由により、成業の見込みがないと認められる者
(5)　休学期間満了後、所定の手続をしない者
(6)　入学料の免除若しくは徴収猶予を不許可とされた者又は入学料の一部免除若しくは徴収猶予を許可された者で、所定の期日までに納付すべき入学料を納付しなかった者
(7)　授業料の納付を怠り、督促してもなお納付しない者
(8)　卒業に要する最終学年を除く1学年の修得単位(第17条第3項により認定された単位は除く。以下この号及び次項において同じ。)が16単位未満の者。ただし、医学部医学科にあっては、第1年次の修得単位が16単位未満の者に限る。"""
print("----")
print("質問: ", instruction)
print("コンテキストなし応答: ", qa2(tokenizer, model, instruction))
print("")
print("コンテキストあり応答: ", qa2(tokenizer, model, instruction, context))
print("\n")

instruction = "半年で15単位しか取れませんでした。除籍されますか？　理由も教えてください。"
print("----")
print("質問: ", instruction)
print("コンテキストなし応答: ", qa2(tokenizer, model, instruction))
print("")
print("コンテキストあり応答: ", qa2(tokenizer, model, instruction, context))
print("\n")
