from data import train_data, test_data
import numpy as np
from numpy.random import randn
import warnings

warnings.filterwarnings("ignore")

# Створення словнику
vocab = list(set([word for text in train_data.keys() for word in text.split()]))
vocab_size = len(vocab)
print(f"{vocab_size} unique words in the training data")

# Призначення індексу кожному слову
word_to_idx = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for i, word in enumerate(vocab)}
print(word_to_idx)
print(index_to_word)

# Повертає масив унітарних векторів,
# які представляють слова у введеному рядку тексту
def create_inputs(text):
    inputs = []
    for w in text.split(' '):
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)
    return inputs

# Застосування функції Softmax для вхідного масиву
def softmax(xs):
    return np.exp(xs) / sum(np.exp(xs))

# Повертає втрати RNN та акуратність для наданих даних
# backdrop визначає, чи потрібно запустити зворотню фазу
def process_data(data, rnn, backprop=True):
    items = list(data.items())
    np.random.shuffle(items)

    loss = 0
    num_correct = 0

    # Цикл для кожного прикладу тренування
    for x, y in items:
        inputs = create_inputs(x)
        target = int(y)

        # Пряме розподілення
        out, _ = rnn.forward(inputs)
        probs = softmax(out)

        # Обчислення втрат / точності
        loss -= float(np.log(probs[target]))
        num_correct += int(np.argmax(probs) == target)

        if backprop:
            d_L_d_y = probs
            d_L_d_y[target] -= 1

            # Зворотне розподілення
            rnn.backprop(d_L_d_y)
    return loss / len(data), num_correct / len(data)

class RNN:
    def __init__(self, input_size, output_size, hidden_size=64):
        # Вага
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000

        # Зміщення
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

        self.last_inputs = None
        self.last_hs = None

    # Виконання передачі нейронної мережі за допомогою вхідних даних
    # Повернення результатів виведення та прихованого стану
    def forward(self, inputs):
        h = np.zeros((self.Whh.shape[0], 1))
        self.last_inputs = inputs
        self.last_hs = {0: h}

        # Виконання кожного кроку в нейронній мережі RNN
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i + 1] = h

        # Compute the output
        y = self.Why @ h + self.by
        return y, h

    # Виконання фази зворотного розповсюдження RNN
    def backprop(self, d_y, learn_rate=2e-2):
        n = len(self.last_inputs)

        d_Why = d_y @ self.last_hs[n].T
        d_by = d_y

        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        d_h = self.Why.T @ d_y

        # Зворотне розповсюдження по часу
        for t in reversed(range(n)):
            temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)

            d_bh += temp
            d_Whh += temp @ self.last_hs[t].T
            d_Wxh += temp @ self.last_inputs[t].T

            d_h = self.Whh @ temp

        # Відсікаємо, щоб попередити розрив градієнтів
        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)

        # Оновлюємо ваги і зміщення з використанням градієнтного спуску
        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by


if __name__ == "__main__":
    # Ініціалізація нашої рекурентної нейронної мережі RNN
    rnn = RNN(vocab_size, 2)

    # Цикл тренування
    for epoch in range(1000):
        train_loss, train_acc = process_data(train_data, rnn, backprop=True)

        if epoch % 100 == 99:
            print(f"Epoch {epoch + 1}")
            print(f"Train loss: {train_loss:0.3f}, Train accuracy: {train_acc:0.3f}")

            test_loss, test_acc = process_data(test_data, rnn, backprop=False)
            print(f"Test loss: {test_loss:.3f}, Test accuracy: {test_acc:.3f}\n")