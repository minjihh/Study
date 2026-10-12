import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

path = './_data/앙상블용/'

def to_num(s):
    if s.dtype != object:
        return s
    s = (s.astype(str).str.replace('▼', '-', regex=False)
           .str.replace('▲', '', regex=False)
           .str.replace(',', '', regex=False).str.strip())
    return pd.to_numeric(s, errors='coerce')

from pandas.api.types import is_numeric_dtype

def load(fname):
    df = pd.read_csv(path + fname, index_col=0, encoding='cp949', thousands=',')

    # 화살표 기호만 들어 있는 컬럼 드랍
    arrow_cols = [c for c in df.columns
                  if not is_numeric_dtype(df[c])
                  and df[c].astype(str).str.contains('[▲▼↑↓]').any()]
    df = df.drop(columns=arrow_cols)

    # 남은 문자열 컬럼 숫자로 변환
    for c in df.columns:
        if not is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '').str.strip(),
                                  errors='coerce')

    df = df.dropna(axis=1, how='all')
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


amore   = load('아모레261012.csv')
hynix   = load('하이닉스261012.csv')     # 파일명 맞게 수정
samsung = load('삼성전자261012.csv')

# 날짜 기준으로 합치기 (세 종목 모두 있는 날짜만)
merged = pd.concat([amore.add_prefix('아모레_'),
                    hynix.add_prefix('하이닉스_'),
                    samsung[['시가']].rename(columns={'시가': '삼성_시가'})],
                   axis=1, join='inner')

# 1) 같은 날 기준 상관
same_day = merged.corr()['삼성_시가'].drop('삼성_시가')

# 2) 전날 데이터 → 다음날 삼성 시가 (실제 예측 상황)
lagged = merged.drop(columns='삼성_시가').shift(1)
lagged['삼성_시가'] = merged['삼성_시가']
next_day = lagged.corr()['삼성_시가'].drop('삼성_시가')

result = pd.DataFrame({'같은날': same_day, '전날→다음날': next_day})
result = result.reindex(result['전날→다음날'].abs().sort_values(ascending=False).index)
print(result.round(3))

plt.figure(figsize=(6, 10))
sns.heatmap(result, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('삼성전자 시가와의 상관')
plt.tight_layout()
plt.show()