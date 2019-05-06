from preprocessing import PreProcessing
from lfmodel import LFModel


def main():
    # data pre-processing
    print('********** Data Preprocessing **********')
    data = PreProcessing()
    data.split_data(random=True, percentage=0.8)
    print('********** Latent Factor Model **********')
    model = LFModel(k=20, epochs=50, learning_rate=0.006, lambda_r=0.06)
    model.train()
    pass


if __name__ == '__main__':
    main()
