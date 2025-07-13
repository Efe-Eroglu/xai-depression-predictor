import pandas as pd

def encode_user_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Streamlit formundan gelen kullanıcı girdisini modele uygun şekilde encode eder.

    Args:
        df (pd.DataFrame): Kullanıcı girdisini içeren tek satırlık DataFrame

    Returns:
        pd.DataFrame: Modele uygun hale getirilmiş, sıralanmış ve eksiksiz DataFrame
    """

    # Dummy sütunlar (eğitim sırasında oluşanlar)
    degree_columns = [
        'Degree_B.Com', 'Degree_B.Ed', 'Degree_B.Pharm', 'Degree_B.Tech',
        'Degree_BA', 'Degree_BBA', 'Degree_BCA', 'Degree_BE', 'Degree_BHM',
        'Degree_BSc', 'Degree_Class 12', 'Degree_LLB', 'Degree_LLM', 'Degree_M.Com',
        'Degree_M.Ed', 'Degree_M.Pharm', 'Degree_M.Tech', 'Degree_MA',
        'Degree_MBA', 'Degree_MBBS', 'Degree_MCA', 'Degree_MD', 'Degree_ME',
        'Degree_MHM', 'Degree_MSc', 'Degree_PhD'
    ]

    # Tüm input birleşimi (manuel çünkü sadece bir kullanıcı satırı var)
    for col in degree_columns:
        df[col] = False

    # Örnek olarak varsayılan bir derece işaretlenebilir
    df['Degree_B.Tech'] = True  # kullanıcıdan alınmıyor şu an

    # Sıralama: Eğitim setindeki sıraya göre sıralanmalı
    column_order = [
        'Gender', 'Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
        'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration',
        'Dietary Habits', 'Work/Study Hours', 'Financial Stress',
        'Family History of Mental Illness'
    ] + degree_columns + ['Suicidal Thoughts']

    # Gerekiyorsa eksik kolonları 0 ile doldur
    for col in column_order:
        if col not in df.columns:
            df[col] = 0

    # Sütun sıralaması
    df = df[column_order]

    return df
