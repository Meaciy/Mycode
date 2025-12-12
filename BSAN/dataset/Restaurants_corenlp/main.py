import matplotlib.pyplot as plt
import seaborn as sns

# 假设输入序列的单词
input_tokens = ['token1', 'token2', 'token3']

# 注意力得分数据，假设为一个二维数组
attention_scores = [[0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9]]

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(attention_scores, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=.5, 
            xticklabels=input_tokens, yticklabels=input_tokens)
plt.xlabel('Input Tokens')
plt.ylabel('Output Tokens')
plt.title('Self-Attention Heatmap')
plt.show()