import os

command = "python test.py --checkpoint ../checkpoints --arithmetic --data ../data/anki_parallel/eng_valid_sel.txt --map-model models/{} --output texts/{}/validsel_eng2spa_{}.txt"

TO_RUNS = (  # signature of models to run
    'n3_lmse_u128_asigmoid_e77_b64_d0.01',
    'n3_lmse_u128_asigmoid_e60_b64_d0.025',
    'n3_lmse_u128_asigmoid_e58_b64_d0.05',
    'n3_lmse_u128_asigmoid_e60_b64_d0.1',
    'n3_lmse_u128_asigmoid_e60_b64_d0.25',
    'n3_lmse_u128_asigmoid_e30_b64_d0.5',
    'n3_lmse_u128_asigmoid_e20_b64_d0.75',
    'n3_lmse_u128_asigmoid_e16_b64_d0.9',
    'n3_lmse_u128_asigmoid_e15_b64_d1.0',
)
RUN_NAME = "shrinking"

os.makedirs(f'texts/{RUN_NAME}', exist_ok=True)

for modelName in os.listdir('./models'):
    if modelName[6:-3] in TO_RUNS:
        print(modelName)
        signature = modelName[6:-3]
        os.system(command.format(modelName, RUN_NAME, signature))
        print("\n\n\n")

print("Text generation complete.")
