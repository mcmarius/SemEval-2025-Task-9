import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times"
})

sns.set_theme()
sns.set_context('paper', font_scale=2.8)

ox = np.arange(1, 11)

# label_set train
f1h  = [0.4257, 0.5166, 0.5886, 0.6012, 0.6180, 0.6370, 0.6526, 0.6528, 0.6876, 0.6997]
f1p1 = [0.2528, 0.3369, 0.3769, 0.4065, 0.4349, 0.4561, 0.4747, 0.4871, 0.4998, 0.5091]
sc1  = [0.3453, 0.4303, 0.4827, 0.5029, 0.5236, 0.5462, 0.5644, 0.5714, 0.5939, 0.6048]

# label_set train_valid
f1h  = [0.4257, 0.5166, 0.5886, 0.6012, 0.6180, 0.6370, 0.6526, 0.6528, 0.6876, 0.6997]
f1p2 = [0.2555, 0.3413, 0.3848, 0.4121, 0.4447, 0.4665, 0.4844, 0.4998, 0.5140, 0.5245]
sc2  = [0.3472, 0.4320, 0.4874, 0.5053, 0.5281, 0.5510, 0.5696, 0.5783, 0.6103, 0.6130]

# label_set train_valid_test -> same as label_set train_valid
# f1h  = [0.4257, 0.5166, 0.5886, 0.6012, 0.6180, 0.6370, 0.6526, 0.6528, 0.6876, 0.6997]
# f1p3 = [0.2555, 0.3413, 0., 0., 0., 0., 0., 0., 0., 0.]
# sc3  = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]

# label_set test
f1h4 = [0.4420, 0.5303, 0.6031, 0.6257, 0.6417, 0.6575, 0.6717, 0.6947, 0.7136, 0.7238]
f1p4 = [0.4832, 0.5912, 0.6543, 0.7135, 0.7369, 0.7602, 0.7815, 0.7935, 0.8035, 0.8121]
sc4  = [0.4504, 0.5541, 0.6234, 0.6607, 0.6828, 0.7037, 0.7183, 0.7368, 0.7530, 0.7628]

def make_plot(oy1, oy2, oy3):
  plt.figure(figsize=(10,6))
  sns.lineplot(x=ox, y=oy1, marker='o', label='F1 Hazard', linewidth=3)
  sns.lineplot(x=ox, y=oy2, marker='s', label='F1 Product', linewidth=3)
  sns.lineplot(x=ox, y=oy3, marker='^', label='Total score', linewidth=3)
  plt.xlabel('k')
  plt.ylabel('Score')
  plt.title('Scores if true label is in top k predictions')
  # 0 is the same as 1 if we do not sort
  # plt.xticks([1, 3, 5, 10]) # set ticks for x-axis
  # plt.show()
  return plt

p1 = make_plot(f1h, f1p1, sc1)
p1.savefig('train.pdf', bbox_inches='tight', pad_inches=0.05)
p2 = make_plot(f1h4, f1p4, sc4)
p2.savefig('test.pdf', bbox_inches='tight', pad_inches=0.05)
# make_plot(f1h, f1p1, sc1)
# make_plot(f1h4, f1p4, sc4)