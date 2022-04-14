import os

command = "python test.py --checkpoint ../checkpoints --arithmetic --data ../data/anki_parallel/eng_test.txt --map-model models/{} --output texts/{}/test_eng2spa_{}.txt"

TO_RUNS = (  # signature of models to run
    'n2_lmse_u128_arelu_e150_b64_d0.1',
    'n1_lmse_u128_arelu_e150_b64_d0.1',
    'n1_lmse_u128_arelu_e300_b64_d0.05',
    'n1_lmse_u128_arelu_e600_b64_d0.025'
)
RUN_NAME = "model_simplification"

os.makedirs(f'texts/{RUN_NAME}', exist_ok=True)

for modelName in os.listdir('./models'):
    if modelName[6:-3] in TO_RUNS:
        print(modelName)
        signature = modelName[6:-3]
        os.system(command.format(modelName, RUN_NAME, signature))
        print("\n\n\n")

print("Text generation complete.")
