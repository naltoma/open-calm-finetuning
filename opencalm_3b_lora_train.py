# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import os

# 生成されたモデルを保存する
SAVE = True

# 生成されたモデルをHugging FaceにPushする
# Trueにした場合は、実行前に下の「Huggingfaceにログイン」のTokenを入力してLoginを押してください。
PUSH_TO_HF = False

# 開発版のtransformers/prftを使う
USE_TRANSFORMERS_DEV = False

# ベースのモデルのhuggingfaceのrepo名
#BASE_MODEL_NAME = "cyberagent/open-calm-7b"
BASE_MODEL_NAME = "cyberagent/open-calm-3b"

LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 3e-4

WARMUP_STEPS = 200
MAX_STEPS = -1
#MAX_STEPS = 5
EVAL_STEPS = 500

# Fine tuningした後のモデル名 == ディレクトリ名
TRAINED_MODEL_NAME = "open-calm-3b-ft"
print(TRAINED_MODEL_NAME)

# モデルを保存するためのディレクトリ
#PEFT_MODEL_PATH = f"/content/drive/MyDrive/Developments/{MODEL_NAME}"
PEFT_MODEL_PATH = f"./{TRAINED_MODEL_NAME}"


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME, 
    torch_dtype=torch.float16,
    device_map='auto',
    offload_folder="./offload"
)

# %% [markdown]
# ## Fine tuningで使う項目をfp32にする

# %%
from torch import nn

for param in model.parameters():
  param.requires_grad = False
  if param.ndim == 1:
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)

print(model)

# embed_outがないと言われる場合は、下記のモデル情報を見て最後の層の名前を指定してみてください。
model.lm_head = CastOutputToFloat(model.embed_out)

# %% [markdown]
# # 教師データとテストデータの作成

# %% [markdown]
# ## データから学習用プロンプトを生成する

# %%
"""
input/instruction/outputからpromptを作る
"""
def generate_prompt(instruction, input=None, response=None):
  def add_escape(text):
    return text.replace('### Response', '###  Response')

  if input:
    prompt = f"""
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{add_escape(instruction.strip())}

### Input:
{add_escape(input.strip())}
""".strip()
  else:
    prompt = f"""
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{add_escape(instruction.strip())}
""".strip()

  if response:
    prompt += f"\n\n### Response:\n{add_escape(response.strip())}<|endoftext|>"
  else:
    prompt += f"\n\n### Response:\n"

  return prompt

def get_response(text):
  marker = f"### Response:\n"
  pos = text.find(marker)
  if pos == -1:  # marker not found
      return None
  return text[pos + len(marker):].strip()

# %% [markdown]
# ## データセットを読み込む

# %%
from datasets import DatasetDict, load_dataset, concatenate_datasets

"""
input/instruction/outputからpromptを作ってtokenizeする
"""
def tokenize_qa_function(sample):
  context = sample.get('input', '').strip()
  instruction = sample.get('instruction', '').strip()
  output = sample.get('output', '').strip()

  prompt = generate_prompt(instruction, context, output)
  return tokenizer(prompt)

"""
nameのdatasetを読み込み、alpaca形式のプロンプトにしてtokenizeする
trainはtest_sizeでtestとsplitする
"""
def process_qa_dataset(name, test_size):
  TOKENIZED_COLUMNS = ['input_ids', 'attention_mask']
  data = load_dataset(name)
  data = data['train'].train_test_split(test_size=test_size)
  remove_columns = [item for item in data['train'].column_names if item not in TOKENIZED_COLUMNS]
  data = data.map(tokenize_qa_function, remove_columns=remove_columns)
  return data


# Q&Aのデータセットを読み込み
datasets = []
datasets.append(process_qa_dataset("kunishou/hh-rlhf-49k-ja", 0.01))
datasets.append(process_qa_dataset("kunishou/databricks-dolly-15k-ja", 0.03))

# 読み込んだdatasetを統合
datasetDict = DatasetDict()
for key in ['train', 'test']:
  datasetDict[key] = concatenate_datasets(list(map(lambda x: x[key], datasets)))
  print(f"{key}: {len(datasetDict[key])}")


# トークンサイズを超えるものは削除
datasetDict = datasetDict.filter(lambda x: len(x['input_ids']) < model.config.max_position_embeddings)

# %%
# 
print(f"{datasets[0]['train'][0]=}")
print(f"{tokenizer.decode(datasets[0]['train'][0]['input_ids'])=}")

# %% [markdown]
# # Fine tuning実行

# %% [markdown]
# ## PEFTのLoRAを設定
# 
# 使うモデルによって、`target_modules`を設定する必要があります。
# 
# GPT-NeoXでは`["query_key_value"]`となり、llamaでは`["q_proj", "v_proj"]`となるようです。
# 
# r, alpha, dropoutがハイパーパラメータです。

# %%
from transformers import TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    bias="none",
    fan_in_fan_out=False,
    target_modules=["query_key_value"],
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# %% [markdown]
# ### [未使用] BLEUによりメトリクス
# 
# 生成された文章の類似度をBLEUで計算する

# %%
"""
from transformers import Trainer, TrainingArguments
from sacrebleu import corpus_bleu

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    decoded_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
    bleu_score = corpus_bleu(decoded_preds, [decoded_labels]).score
    print(bleu_score)
    return {'bleu_score': bleu_score}
"""

# %% [markdown]
# ## 学習実行

# %%
import transformers
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# 学習の設定
trainer = Trainer(
    model=model, 
    train_dataset=datasetDict['train'],
    eval_dataset=datasetDict['test'],
    args=TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE, 
        gradient_accumulation_steps=BATCH_SIZE // MICRO_BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE, 
        fp16=True,
        num_train_epochs=EPOCHS,
        save_strategy='epoch',
        output_dir="result",
        evaluation_strategy='steps',
        eval_steps=EVAL_STEPS,
        logging_dir='./logs',
        logging_steps=100,
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    # compute_metrics=compute_metrics
)

# 学習開始
model.config.use_cache = False
trainer.train()
model.config.use_cache = True

# 推論モード
model.eval()

# Google Driveに保存
if SAVE:
#  !mkdir -p "{PEFT_MODEL_PATH}"
  if not os.path.isdir(PEFT_MODEL_PATH):
    os.makedirs(PEFT_MODEL_PATH)
  model.save_pretrained(PEFT_MODEL_PATH)

# %% [markdown]
# # テスト実行

# %%
def qa(instruction, context=None):
  prompt = generate_prompt(instruction, context)

  batch = tokenizer(prompt, return_tensors='pt')
  with torch.cuda.amp.autocast():
    output_tokens = model.generate(
        **batch,
        max_new_tokens=256,
        temperature = 0.7,
        repetition_penalty=1.05
    )

  text = tokenizer.decode(output_tokens[0],pad_token_id=tokenizer.pad_token_id,
skip_special_tokens=True)
  return get_response(text)

instruction = "トヨタ自動車は何年設立ですか？"
context = "豊田自動織機製作所自動車部時代は、社名中の「豊田」の読みが「トヨダ」であったため、ロゴや刻印も英語は「TOYODA」であった。エンブレムは漢字の「豊田」を使用していた。しかし、1936年夏に行われた新トヨダマークの公募で、約27,000点の応募作品から選ばれたのは「トヨダ」ではなく「トヨタ」のマークだった。理由として、デザイン的にスマートであること、画数が8画で縁起がいいこと、個人名から離れ社会的存在へと発展することなどが挙げられている[11]。1936年9月25日に「トヨタ（TOYOTA）」の使用が開始され、翌年の自動車部門独立時も「トヨタ自動車工業株式会社」が社名に採用された。"
print("----")
print(instruction)
print(qa(instruction))
print("")
print(qa(instruction, context))
print("\n")

instruction = "マイナンバーカードの受け取りは免許書を持って行けばいいですか？"
context = "マイナンバーカードの受け取りに必要な書類は以下のとおりです。\n交付通知書\n本人確認書類（有効期間内のもの）\n顔写真付きの本人確認書類は1点\nその他は2点（例：健康保険証＋年金手帳）\nお持ちの方のみ\n通知カード\n住民基本台帳カード\nマイナンバーカード\n詳しくは必要な持ち物をご確認ください。"
print("\n----")
print(instruction)
print(qa(instruction))
print("")
print(qa(instruction, context))
print("\n")


instruction = "情報セキュリティ対策に関わる責任者と担当者の役割や権限が明確になっていますか？具体的には、個人情報保護責任者、個人情報保護担当者は任命されていますか？"
context = "個人情報保護管理者: 山形太郎\n責任と権限\n・JISQ15001に適合したPMSを構築、運用する\n・PMSの運用状況や成果を、トップマネジメントに報告する\n----\n個人情報保護監査責任者\n宮崎健吾\n責任と権限\n・内部監査計画書の作成\n・内部監査員の選定\n・内部監査の指揮 ・内部監査報告書の作成及びトップマネジメントへの報告\n"
print("\n----")
print(instruction)
print(qa(instruction))
print("")
print(qa(instruction, context))
print("\n")


instruction = "情報セキュリティ対策に関わる責任者と担当者の役割や権限が明確になっていますか？具体的には、個人情報保護責任者、個人情報保護担当者は任命されていますか？"
context = "個人情報保護管理者: 任命なし\n責任と権限\n・JISQ15001に適合したPMSを構築、運用する\n・PMSの運用状況や成果を、トップマネジメントに報告する\n----\n個人情報保護監査責任者\n宮崎健吾\n責任と権限\n・内部監査計画書の作成\n・内部監査員の選定\n・内部監査の指揮 ・内部監査報告書の作成及びトップマネジメントへの報告\n"
print("\n----")
print(instruction)
print(qa(instruction))
print("")
print(qa(instruction, context))
print("\n")

instruction = "地球温暖化とはなんですか？"
context = """リモートワークでリモートワークができた
毎日でも食べたいということは毎日でも食べているというわけではない
今のままではいけないと思っています。だからこそ日本は今のままではいけないと思っている
約束は守るためにありますから、約束を守るために全力を尽くします"""
print("\n----")
print(instruction)
print(qa(instruction))
print("")
print(qa(instruction, context))
print("\n")


# %% [markdown]
# # Huggingfaceにアップロード
# 上で`PUSH_TO_HF`をTrueにしておいた場合、生成されたモデルをHuggingFaceにて公開する

# %% [markdown]
# ## リポジトリを作る

# %%
if PUSH_TO_HF:
  hub = huggingface_hub.HfApi()
  hub.create_repo(repo_id=MODEL_NAME, exist_ok=True)

# %% [markdown]
# ## モデルをアップする
# tokenizerは親モデルのを使うのでpushする必要なし

# %%
if PUSH_TO_HF:
  model.push_to_hub(MODEL_NAME)


