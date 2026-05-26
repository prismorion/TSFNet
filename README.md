# TSFNet

Библиотека в рамках дипломного проекта на C# для прогнозирования временных рядов с помощью нейронных сетей. Содержит реализации MLP, RNN и GRU, а также инфраструктуру для их обучения и оценки.

<br/>

## Описание библиотеки

- **Модели:** `MLP`, `RNN`, `GRU`. Реализованы с нуля без внешних зависимостей.
- **Буферы:** используются отдельные классы для хранения и перезаписи в них данных для уменьшения количества аллокаций и снижения нагрузки на garbage collector.
- **Интерфейсы и дженерики:** `ITrainable<TInput, TBuffer, TSnapshot>` является общим контрактом для всех моделей. Параметр `TInput` различает форматы входа (`double[]` для MLP, `double[][]` для RNN/GRU), `TBuffer` - тип рабочих буферов, `TSnapshot` - структура для сохранения и восстановления весов.
- **Trainer:** статический класс с методом `Fit`, который принимает любую модель, реализующую `ITrainable`, датасет, гиперпараметры и опции обучения. Берёт на себя всё управление обучением.
- **Предобработка:** библиотека имеет классы `StandardScaler` для стандартизации (Z-score) и `Differencer` для приведения ряда к стационарному виду.

<br/>

## Пример использования

**Создание датасета:**

|       | Один пример  | Весь набор     | Датасет               |
|-------|--------------|----------------|-----------------------|
| MLP   | `double[]`   | `double[][]`   | `Dataset<double[]>`   |
|RNN/GRU| `double[][]` | `double[][][]` | `Dataset<double[][]>` |

```C#
double[][][] inputs = ...;   // окна признаков
double[][] targets = ...;  // ожидаемые значения
var dataset = new Dataset<double[][]>(inputs, targets);
```

<br/>

**Создание модели:**

```C#
var gru = new GRU(1, 32, 1);
```

<br/>

**Гиперпараметры:**

```C#
Hyperparameters hyperparameters = new Hyperparameters();
hyperparameters.learningRate = 0.01;
hyperparameters.batchSize = 2;
hyperparameters.l2Lambda = 0.01;
hyperparameters.threshold = 5;
```

<br/>

**Опции:**

```C#
TrainingOptions trainingOptions = new TrainingOptions();
trainingOptions.epochs = 100;
trainingOptions.reportEvery = 10;
```

<br/>

**Обучение:**

```C#
FitResponse GRUResponse = Trainer.Fit(gru, dataset, hyperparameters, trainingOptions);
```

- Метод Fit возвращает response с логами обучения

<br/>

**Пример работы обученных моделей:**

<br/>

<img width="732" height="491" alt="image" src="https://github.com/user-attachments/assets/7a3770e4-9c56-4bbb-9365-76e38d1633c3" />

<br/>

Применена предобработка данных по Z-score и вывод результатов на график.

<br/>


## План доработок

- Добавить авторегрессию.
- Добавить описание всех методов через summary.
- Настроить в методах проверки корректности входных данных.
