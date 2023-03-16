import os

command = "python test.py --checkpoint ../checkpoints --arithmetic --data ../data/anki_parallel/eng_new_sel.txt --map-model models/{} --output texts/{}/newsel_eng2spa_{}.txt"

TO_RUNS = (  # signature of models to run
    'n2_lmse_u256_arelu_e200_b64_d0.1',
    #'n3_lmse_u256_arelu_e200_b64_d0.1',
    #'n2_lmse_u128_arelu_e200_b64_d0.1',
    'n1_lmse_u128_arelu_e200_b64_d0.1'
)
RUN_NAME = "writeup"

os.makedirs(f'texts/{RUN_NAME}', exist_ok=True)

for modelName in os.listdir('./models'):
    if modelName[6:-3] in TO_RUNS:
        print(modelName)
        signature = modelName[6:-3]
        os.system(command.format(modelName, RUN_NAME, signature))
        print("\n\n\n")

print("Text generation complete.")
