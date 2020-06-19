import numpy as np


def normalize(data):
    minimums = [abs(np.min(data))]
    data += minimums[-1]
    maximums = [np.max(data)]
    data /= maximums[-1]
    return data, minimums, maximums


def reverse_normalize(data, minimums, maximums):
    data = (maximums[0] * data) - minimums[0]
    return data


def normalize_vae(data):
    minimums = []
    maximums = []

    first_data = data[:, :24]
    minimums.append(abs(np.min(first_data)))
    first_data += minimums[-1]
    maximums.append(np.max(first_data))
    first_data /= maximums[-1]
    data[:, :24] = first_data

    first_data = data[:, 24:48]
    minimums.append(abs(np.min(first_data)))
    first_data += minimums[-1]
    maximums.append(np.max(first_data))
    first_data /= maximums[-1]
    data[:, 24:48] = first_data

    first_data = data[:, 48:52]
    minimums.append(abs(np.min(first_data)))
    first_data += minimums[-1]
    maximums.append(np.max(first_data))
    first_data /= maximums[-1]
    data[:, 48:52] = first_data

    first_data = data[:, 52]
    minimums.append(abs(np.min(first_data)))
    first_data += minimums[-1]
    maximums.append(np.max(first_data))
    first_data /= maximums[-1]
    data[:, 52] = first_data

    first_data = data[:, 53]
    minimums.append(abs(np.min(first_data)))
    first_data += minimums[-1]
    maximums.append(np.max(first_data))
    first_data /= maximums[-1]
    data[:, 53] = first_data
    return data, minimums, maximums


def reverse_normalize_vae(data, minimums, maximums):
    data[:, :24] = (maximums[0] * data[:, :24]) - minimums[0]
    data[:, 24:48] = (maximums[1] * data[:, 24:48]) - minimums[1]
    data[:, 48:52] = (maximums[2] * data[:, 48:52]) - minimums[2]
    data[:, 52] = (maximums[3] * data[:, 52]) - minimums[3]
    data[:, 53] = (maximums[4] * data[:, 53]) - minimums[4]
    return data

def validate(a):
    return