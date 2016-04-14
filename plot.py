import matplotlib.pyplot as plt


def plot_by_column(expected_output, predicted_output, sub_num=11, begin=0,
                 time_steps=168, caption='Load by time for zone: '):
    if sub_num > len(predicted_output[0]):
        sub_num = len(predicted_output[0])
    row = -(-sub_num // 3)
    plt.figure(figsize=(16, row * 4))
    for i in range(len(predicted_output[0])):
        index = i + 1
        plt.subplot(row, 3, index)
        plt.plot(expected_output[begin:begin+time_steps, i])
        plt.plot(predicted_output[begin:begin+time_steps, i])
        plt.title(caption + str(index))
    plt.show()


def plot_by_row(expected_output, predicted_output, sub_num=20,
                caption='Load by zone for hour: '):
    if sub_num > len(expected_output):
        sub_num = len(expected_output)
    row = -(-sub_num // 3)
    plt.figure(figsize=(16, row * 4))
    for i in range(sub_num):
        index = i + 1
        plt.subplot(row, 3, index)
        plt.plot(expected_output[i])
        plt.plot(predicted_output[i])
        plt.title(caption + str(index))
    plt.show()
