def test_concatentate_lists():
    lists = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    expected_output = [1, 2, 3, 4, 5, 6]
    assert(concatenate_lists(lists) == expected_output)


def concatenate_lists(lists):
    result = []
    for single_list in lists:
        result.extend(single_list)
    return result


def test_convert_mnist_images_to_lists():
    mnist_images = [
        [
            [1, 2, 3],
            [4, 5, 6]
        ],
        [
            [7, 8, 9],
            [10, 11, 12]
        ]
    ]

    expected_output = [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12]
    ]

    assert(convert_mnist_images_to_lists(mnist_images) == expected_output)


def convert_mnist_images_to_lists(mnist_images):
    converted_images = []

    for mnist_image in mnist_images:
        converted_images.append(concatenate_lists(mnist_image))

    return converted_images


def test_convert_number_to_decimal_output():
    label = 1
    expected_output = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert(convert_number_to_decimal_output(label) == expected_output)

    label = 2
    expected_output = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    assert(convert_number_to_decimal_output(label) == expected_output)

    label = 3
    expected_output = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    assert(convert_number_to_decimal_output(label) == expected_output)

    label = 4
    expected_output = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    assert(convert_number_to_decimal_output(label) == expected_output)

    label = 5
    expected_output = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    assert(convert_number_to_decimal_output(label) == expected_output)

    label = 6
    expected_output = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    assert(convert_number_to_decimal_output(label) == expected_output)

    label = 7
    expected_output = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    assert(convert_number_to_decimal_output(label) == expected_output)

    label = 8
    expected_output = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    assert(convert_number_to_decimal_output(label) == expected_output)

    label = 9
    expected_output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert(convert_number_to_decimal_output(label) == expected_output)

    label = 0
    expected_output = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert(convert_number_to_decimal_output(label) == expected_output)


def convert_number_to_decimal_output(number):
    output = [0] * 10

    output[number] = 1

    return output


def test_create_zeros_matrix():
    x = 5
    y = 3
    expected_output = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]

    assert(create_zeros_matrix(x, y) == expected_output)


def create_zeros_matrix(width, height):
    matrix = []

    for _ in range(height):
        matrix.append([0] * width)

    return matrix


if __name__ == "__main__":
    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    raw_mnist_images = x_train[0:1000]
    mnist_images = convert_mnist_images_to_lists(raw_mnist_images)
    mnist_labels = y_train[0:1000]
    correct_outputs = []

    for mnist_label in list(mnist_labels):
        correct_outputs.append(
            convert_number_to_decimal_output(mnist_label))

    weights = create_zeros_matrix(
        len(mnist_images[0]), len(correct_outputs[0]))
