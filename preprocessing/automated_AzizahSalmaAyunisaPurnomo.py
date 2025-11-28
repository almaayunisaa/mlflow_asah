# File Automate.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from joblib import dump

def automate_Azizah(df, save_path, file_path):
  df.drop('Sl. No', axis=1, inplace=True)
  df.drop('Patient File No.', axis=1, inplace=True)
  df.drop('Unnamed: 44', axis=1, inplace=True)
  
  df.drop_duplicates(inplace=True)
  df.dropna(inplace=True)

  df['AMH(ng/mL)'] = pd.to_numeric(df['AMH(ng/mL)'], errors='coerce')
  df['II    beta-HCG(mIU/mL)'] = pd.to_numeric(df['II    beta-HCG(mIU/mL)'], errors='coerce')
  df.dropna(inplace=True)
  
  y = df['PCOS (Y/N)']
  X = df.drop('PCOS (Y/N)', axis=1)
  
  num_cols =  X.select_dtypes(include=['number']).columns
  
  Q1 = X[num_cols].quantile(0.25)
  Q3 = X[num_cols].quantile(0.75)
  IQR = Q3 - Q1

  condition = (X[num_cols] < (Q1 - 1.5 * IQR)) | (X[num_cols] > (Q3 + 1.5 * IQR))
  X[num_cols] = X[num_cols].mask(condition)

  # Pipeline untuk fitur numerik
  numeric_transformer = Pipeline(steps=[
       ('scaler', StandardScaler())
  ])

   # Column Transformer
  preprocessor = ColumnTransformer(
      transformers=[
            ('num', numeric_transformer, num_cols)
      ]
  )

  # scaler = StandardScaler()
  # df[num_cols] = scaler.fit_transform(df[df.select_dtypes(include='number').columns])
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Fitting dan transformasi data pada training set
  X_train = preprocessor.fit_transform(X_train)
  # Transformasi data pada testing set
  X_test = preprocessor.transform(X_test)
  # Simpan pipeline
  dump(preprocessor, save_path)
  
  # Mendapatkan nama kolom tanpa kolom target
  column_names = X.columns

  # Membuat DataFrame kosong dengan nama kolom
  df_header = pd.DataFrame(columns=column_names)

  # Menyimpan nama kolom sebagai header tanpa data
  df_header.to_csv(file_path, index=False)

  print(f"Nama kolom berhasil disimpan ke: {file_path}")

  return X_train, X_test, y_train, y_test

df = pd.read_csv("../PCOS_raw.csv")
save_path = "preprocessing.joblib"
file_path = "PCOS_preprocessing.csv"
automate_Azizah(df, save_path, file_path)