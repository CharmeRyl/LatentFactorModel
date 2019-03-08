from preprocessing import PreProcessing


def main():
    # data pre-processing
    print('********** Data Preprocessing **********')
    data = PreProcessing()
    data.split_data(random=True, percentage=0.8)

    pass


if __name__ == '__main__':
    main()
