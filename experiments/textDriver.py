import os

command = "python test.py --checkpoint ../checkpoints --arithmetic --data ../data/anki_parallel/eng_valid_sel.txt --map-model models/{} --output texts/validsel_eng2spa_{}.txt"

for modelName in os.listdir('./models'):
    print(modelName)
    signature = modelName[6:-3]
    os.system(command.format(modelName, signature))
    print("\n\n\n")
