ORIGINAL_DIM = 53
LATENT_DIM = 100
STRUCTURE_ENCODER = [[128, 'relu'],
                     [256, 'relu'],
                     [384, 'relu']]
STRUCTURE_DECODER = [[384, 'relu'],
                     [256, 'relu'],
                     [128, 'relu']]
EPOCHS = 1
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
