# данная переменная отвечает за размер входящего вектора в vae
ORIGINAL_DIM = 53
# данная переменная отвечает за размер слоя внутреннего представления
LATENT_DIM = 100
# данная переменная отвечает за структуру энкодера, в каждой ячейке лежит количество нейронов и функция активации для соответствующего слоя
STRUCTURE_ENCODER = [[128, 'relu'],
                     [256, 'relu'],
                     [384, 'relu']]
# данная переменная отвечает за структуру декодера, в каждой ячейке лежит количество нейронов и функция активации для соответствующего слоя
STRUCTURE_DECODER = [[384, 'relu'],
                     [256, 'relu'],
                     [128, 'relu']]
# данная переменная отвечает за количество эпох при обучении vae
EPOCHS = 1
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
