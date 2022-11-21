from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from main import VideoAnalyzer
import pandas as pd

v = VideoAnalyzer()

df = pd.read_csv('clips_events.csv',low_memory=False)
df['event'] = df['event'].astype('string')
df['path'] = df['path'].astype('string')

df['event'] = np.where(
    df['event'].str.contains('pass', regex=False),
    1,
    0
)

def get_trace(video_path):
    return v.prepare(v.run(video_path)).values.tolist()

print('working')
df['trace'] = df['path'].apply(get_trace, convert_dtype=True)

print('done')
model = Sequential()
model.add(LSTM((1), batch_input_shape=(None, 101, 5), return_sequences=False))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
model.summary()

x_train = list(df.trace)
y_train = list(df.event)
x_test = list(df.trace)
y_test = list(df.event)

history = model.fit(x_train, y_train, epochs=400, validation_data=(x_test, y_test))
results = model.predict(x_test)

plt.scatter(range(3), results, c='r')
plt.scatter(range(3), y_test, c='g')
plt.show()

plt.plot(history.history['loss'])
plt.show()

model.save('VideoPredictionModel')