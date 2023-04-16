import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob


file_path = 'C:/Users/nicho/Desktop/SUTD Term 6/50.021/Project/AI_DOOM/logs/'
file_ls = glob(file_path + '*test_results.csv')
print(file_ls)

result_df = pd.DataFrame([])
temp_df = pd.DataFrame([])
for file in file_ls:
    temp_df = pd.read_csv(file).reset_index(drop=True)
    temp_df = temp_df.groupby('file_name').mean().reset_index()
    result_df = pd.concat([result_df, temp_df])
result_df = result_df.rename(columns={'file_name': 'exp_name'})
result_df['exp_name'] = result_df['exp_name'].apply(lambda name: name[37:-20])
result_df['exp_name'] = result_df['exp_name'].apply(lambda name: "0" + name if ("frame_repeat" in name and len(name) != 15) else name)
# result_df['exp_name'] = result_df['exp_name'].apply(lambda name: '12_frame_repeat(original)' if name == 'v0' else name)
result_df['exp_name'] = result_df['exp_name'].apply(lambda name: 'original' if name == 'v0' else name)
result_df = result_df.sort_values(by=['exp_name'])
print(result_df)
result_df.to_csv("results.csv")
plt.figure(figsize=(10,6))
sns.barplot(x = result_df.exp_name, y = result_df.test_results)
plt.title("DuelQN Experiment Results for Extra Rewards", fontsize=18)
plt.xlabel("Experiment", fontsize=16)
plt.ylabel("Average Score across 25 Iters", fontsize=16)
plt.xticks(rotation = 90)
plt.savefig("C:/Users/nicho/Desktop/SUTD Term 6/50.021/Project/AI_DOOM/plots/" +"Experiment Results.jpg")