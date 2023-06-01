# 学科サーバで、OpenCALM-3BをLoRAでFine tuningして対話ができるようにしてみる
## 元ネタ
- [OpenCALM-7BをLoRAでFine tuningして対話ができるようにする](https://note.com/masuidrive/n/n0e2a11fc5bfa)
- 参考
    - [[翻訳] Hugging faceのPEFTのクイックツアー](https://qiita.com/taka_yayoi/items/9196444274d6a63cda76)
    - [LLMを効率的に再学習する手法(PEFT)を解説](https://blog.brainpad.co.jp/entry/2023/05/22/153000)

## 趣旨
- 今回
    - OpenCALMで[PEFT/LoRA](https://huggingface.co/docs/peft/quicktour)を試してみたい。
    - [学科サーバ](https://ie.u-ryukyu.ac.jp/syskan/server_configuration/)のGPUでファインチューニングをどのぐらい動かせそうか検証したい。
        - ちなみにただ動かすだけ（学習なしで応答させるだけ）なら、M1 macbook Air（16GB）で7bも可能。ただし1応答に3時間ぐらいかかるw
    - 最近のtransformersをSingularityで動かす際の手順を再確認したい。
- 次回予定
    - [ggml](https://github.com/ggerganov/ggml)での 4bit 量子化 を試してみたい。

## 前提
- [学科サーバ](https://ie.u-ryukyu.ac.jp/syskan/server_configuration/)を使います。基本的な使い方は[SingularityとSlurmの実践例](https://ie.u-ryukyu.ac.jp/syskan/opening-introduction/singularity-slurm.html#1)を参照。amaneにログインして作業することになります。
- ついでに[キャッシュサーバ](https://ie.u-ryukyu.ac.jp/syskan/internal/cache-server/)も使ってみます。

---
## 手順
### コードの用意
今回は[OpenCALM-7BをLoRAでFine tuningして対話ができるようにする](https://note.com/masuidrive/n/n0e2a11fc5bfa)を参考に、[opencalm_3b_lora_train.py](./opencalm_3b_lora_train.py)を用意しました。7Bを3Bに変更し、不要箇所削除したぐらいだったかな。M1 macで動かすためにあちこち修正したけど、学科サーバで動かす際には元に戻したはず。
- [opencalm_3b_lora_train.py](./opencalm_3b_lora_train.py): open-calm-3bで学習するコード。
- [calm-ft.def](./calm-ft.def): SIFファイル作成用設定ファイル。
- [train.sbatch](./train.sbatch): slurm経由でジョブ実行するためのファイル。
- [utils.py](./utils.py): train.pyから使いまわしたい関数を抜き出し、model, tokenizerを引数で指定するようにしたもの。
- [test_model.py](./test_model.py): 動作確認用のファイル。

---
### SIFファイルの作成
[数年前に試した](https://github.com/naltoma/trial-keras-bert/)際にはbuildでコケることが頻繁にあり、結局shellモードで起動して手動で環境構築したほうが早いみたいな悲しい状態でした。が、今は[huggingfaceがdocker用意してる](https://hub.docker.com/r/huggingface/transformers-pytorch-gpu)ので、これをベースにするとあっさり完成。今回は以下の手順で作成しました。

```shell
# dockerベースファイルをダウンロード
singularity pull docker://huggingface/transformers-pytorch-gpu

# 追加設定を calm-ft.def として作成し、ビルド。
singularity build --fakeroot huggingface.sif calm-ft.def
```

これだけ。pullする部分含めてdefファイルに書いても動きそうですが、動作未確認です。

[calm-ft.def](./calm-ft.def)のpipで、キャッシュサーバ指定しつつ peft accelerate を追加インストールしています。逆に言えばそれ以外は最初から入ってました。助かる。

---
### ジョブ投入用ファイルの作成
huggingfaceのdockerから作成した環境では、``python3``で動かす必要があります。ちなみに動作確認した時点ではPython 3.8.10でした。

[train.sbatch](./train.sbatch)のように、``--nv``のオプション指定を忘れずに。これ書かないとGPU使ってくれません。他の説明は[前回の補足](https://github.com/naltoma/trial-keras-bert/blob/main/tutorial.md#b-どうやってslurmで動かすのか)を見るなりしてください。

---
### ジョブ投入（学習）
```shell
# ジョブ投入
sbatch train.sbatch

# ジョブ管理状況の確認
squeue
```

train.sbatchのlog指定した場所にログが記録されるので、``tail -f logs/open-calm3b-4836.err`` のようにtailコマンド実行すると動作状況を確認できます。今回の例（65kぐらいのデータセットで1 epoch）だと2時間ぐらいかかりました。

---
### 動作確認
PEFTで学習したモデルを保存したファイルの扱いにややトラブりました。

model.save_pretrained()で保存すると、指定したディレクトリに adapter_config.json, adapter_model.bin の2つが保存される。が、ファイルサイズ見るととっても小さい。[公式ページ](https://huggingface.co/docs/peft/quicktour#save-and-load-a-model)参照する限りでは「This only saves the incremental 🤗 PEFT weights」らしく、差分のみ保存してるらしい。
```shell
amane:tnal% du -sh open-calm-3b-ft/*
512	open-calm-3b-ft/adapter_config.json
5.1M	open-calm-3b-ft/adapter_model.bin
```

差分保存ということなので、モデルをロードするためにはまずベースモデルを読み込み、差分を適用するという手順を踏む必要があります。このベース情報も adapter_ter_config.json に保存されるので、こんな感じで読み込みます。
```Python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

peft_model_id = "open-calm-3b-ft"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path) #ベースモデル読み込み
model = PeftModel.from_pretrained(model, peft_model_id) # 差分適用
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
```

これを踏まえて動作確認用に用意したコードが[test_model.py](./test_model.py)。

学習するわけじゃないしということでslurm使わずに直接実行した結果は以下の通り。コンテキストを与えない場合と、与えた場合の2通りの応答文を出力させています。質問2個x2応答=4応答。モデル読み込み自体に1分ぐらい時間かかってて、GPU使った状態での応答は数秒程度なので結構早い。
```shell
amane:tnal% time singularity exec --nv huggingface.sif python3 test_model.py
INFO:    underlay of /usr/bin/nvidia-smi required more than 50 (464) bind mounts
----
質問: 半年で15単位しか取れませんでした。除籍されますか？
Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
コンテキストなし応答:  いいえ、それはあなたにとって良いことです!

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
コンテキストあり応答:  退学処分となります


----
質問: 半年で15単位しか取れませんでした。除籍されますか？　理由も教えてください。
Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
コンテキストなし応答:  確かに、それはあなたがあなたの学士号を取得するのに十分な時間がないことを意味します!

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
コンテキストあり応答:  退学処分となります


singularity exec --nv huggingface.sif python3 test_model.py  595.56s user 151.06s system 975% cpu 1:16.52 total
```

「いいえ、それはあなたにとって良いことです!」はコンテキスト与えなかった場合の応答。
「退学処分となります」は学則を与えた場合の応答。間違った応答してますが、コンテキストありの方が意味が通る応答返していますね。

---
## サマリ
- Tesla V100S (GPU 32GB) 上での学習について
    - 7b＋今回の学習（65kぐらいのデータセット）だと ``torch.cuda.OutOfMemoryError`` となり停止する。ただしバッチサイズあたりの調整でなんとかなる範囲な気もする。
    - 3bだと23GB消費しつつ学習できる。1エポック2時間ぐらい。
- PEFT/LoRAの動かし方と、コンテキストを与えることの影響をある程度観察できた。
    - 特定情報源に基づいて答えさせるには、情報源を抽出してコンテキストとして与えるぐらいでなんとかなる部分もありそう。今回の例では不適切な応答をしているけど、そもそも1 epochしかしていないし、コンテキストが不十分という可能性もある。
- その他
    - 今回は 1 epoch しか動かしていないので、10 epochぐらい回しっぱなしにしてみるか。[LLaMA-Adapter](https://github.com/ZrrSkywalker/LLaMA-Adapter)だと5 epochらしい。
    - [FlexGen](https://github.com/FMInference/FlexGen)みたいにリソース調整できるなら7bでの学習も十分視野に入りそうな気がするな。
