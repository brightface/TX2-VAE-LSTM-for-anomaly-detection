from base import BaseDataGenerator
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig


class DataGenerator(BaseDataGenerator):
  def __init__(self, config):
    super(DataGenerator, self).__init__(config)
    # load data here: generate 3 state variables: train_set, val_set and test_set
    self.load_NAB_dataset(self.config['dataset'],self.config['dataset1'],self.config['dataset2'],self.config['dataset3'], self.config['y_scale'])

  def load_NAB_dataset(self, dataset, dataset1,dataset2,dataset3, y_scale=6):
    data_dir = '../datasets/NAB-known-anomaly/'
    data = np.load(data_dir + dataset + '.npz')

    #custom
    data1 = np.load(data_dir+ dataset1 + '.npz')
    data2 = np.load(data_dir+ dataset2 + '.npz')
    data3 = np.load(data_dir+ dataset3 + '.npz')
    # data4 = np.load(data_dir+ dataset4 + '.npz')
    
    # normalise the dataset by training set mean and std
    train_m = data['train_m']
    train_std = data['train_std']
    readings_normalised = (data['readings'] - train_m) / train_std
    
    # custom normalise
    train_m1 = data1['train_m']
    train_std1 = data1['train_std']
    readings_normalised1 = (data1['readings'] - train_m1) / train_std1

    train_m2 = data2['train_m']
    train_std2 = data2['train_std']
    readings_normalised2 = (data2['readings'] - train_m2) / train_std2

    train_m3 = data3['train_m']
    train_std3 = data3['train_std']
    readings_normalised3 = (data3['readings'] - train_m3) / train_std3

    # train_m4 = data4['train_m']
    # train_std4 = data4['train_std']
    # readings_normalised4 = (data4['readings'] - train_m4) / train_std4


    #custom dataset by traing set mean std
    # plt.subplot(1,2)
    # fig, axs = plt.subplots(1, 1, figsize=(18, 4), edgecolor='k')
    # fig.subplots_adjust(hspace=.4, wspace=.4)
    # axs.plot(data['t'], readings_normalised)
    # if data['idx_split'][0] == 0:
    #   axs.plot(data['idx_split'][1] * np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b-')
    # else:
    #   for i in range(2):
    #     axs.plot(data['idx_split'][i] * np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b-')
    # axs.plot(*np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b--')
    # for j in range(len(data['idx_anomaly'])):
    #   axs.plot(data['idx_anomaly'][j] * np.ones(20), np.linspace(-y_scale, 0.75 * y_scale, 20), 'r--')
    # axs.grid(True)
    # axs.set_xlim(0, len(data['t']))
    # axs.set_ylim(-y_scale, y_scale)
    # axs.set_xlabel("timestamp (every {})".format(data['t_unit']))
    # axs.set_ylabel("readings")
    # axs.set_title("{} dataset\n(normalised by train mean {:.4f} and std {:.4f})".format(dataset, train_m, train_std))
    # axs.legend(('data', 'train test set split', 'anomalies'))
    # savefig(self.config['result_dir'] + '/raw_data_normalised.pdf')
    
    # print(len(data1['training']))
    # print(len(data2['training']))

    # plot normalised data
    # plt.subplot(1,2,2)
    # fig, axs = plt.subplots(1, 1, figsize=(18, 4), edgecolor='k')
    # fig.subplots_adjust(hspace=.4, wspace=.4)
    # axs.plot(data2['t'], readings_normalised2)
    # if data2['idx_split'][0] == 0:
    #   axs.plot(data2['idx_split'][1] * np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b-')
    # else:
    #   for i in range(2):
    #     axs.plot(data2['idx_split'][i] * np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b-')
    # axs.plot(*np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b--')
    # for j in range(len(data2['idx_anomaly'])):
    #   axs.plot(data2['idx_anomaly'][j] * np.ones(20), np.linspace(-y_scale, 0.75 * y_scale, 20), 'r--')
    # axs.grid(True)
    # axs.set_xlim(0, len(data2['t']))
    # axs.set_ylim(-y_scale, y_scale)
    # axs.set_xlabel("timestamp (every {})".format(data2['t_unit']))
    # axs.set_ylabel("readings")
    # axs.set_title("{} dataset\n(normalised by train mean {:.4f} and std {:.4f})".format(dataset2, train_m2, train_std2))
    # axs.legend(('data', 'train test set split', 'anomalies'))
    # savefig(self.config['result_dir'] + '/raw_data_normalised.pdf')
    
    print(len(data1['training']))
    print(len(data2['training']))
    #데이터 숫자가 DATA1이 적으니 1로 맞춰놓고 트레이닝한다.(MACHINE_TEMP>CPU)
    # data2['training'] = data1['training']
    
    # slice training set into rolling windows
    n_train_sample = len(data1['training'])
    print("n_train_sample",n_train_sample)
    n_train_vae = n_train_sample - self.config['l_win'] + 1 #3453개 (3500-48+1)
    # print(n_train_sample)

    rolling_windows = np.zeros((n_train_vae, self.config['l_win']))
    for i in range(n_train_sample - self.config['l_win'] + 1):
      rolling_windows[i] = data['training'][i:i + self.config['l_win']]

    #custom rolling_windows
    rolling_windows1 = np.zeros((n_train_vae, self.config['l_win']))
    for i in range(n_train_sample - self.config['l_win'] + 1):
      rolling_windows1[i] = data['training'][i:i + self.config['l_win']]
    
    rolling_windows2 = np.zeros((n_train_vae, self.config['l_win']))
    for i in range(n_train_sample - self.config['l_win'] + 1):
      rolling_windows2[i] = data2['training'][i:i + self.config['l_win']]
    
    rolling_windows3 = np.zeros((n_train_vae, self.config['l_win']))
    for i in range(n_train_sample - self.config['l_win'] + 1):
      rolling_windows3[i] = data3['training'][i:i + self.config['l_win']]
    
    # rolling_windows4 = np.zeros((n_train_vae, self.config['l_win']))
    # for i in range(n_train_sample - self.config['l_win'] + 1):
    #   rolling_windows4[i] = data4['training'][i:i + self.config['l_win']]



    print(rolling_windows.shape)
    # create VAE training and validation set
    idx_train, idx_val, self.n_train_vae, self.n_val_vae = self.separate_train_and_val_set(n_train_vae)
    #custom create # create VAE training and validation set
    k1 = np.expand_dims(rolling_windows1[idx_train],-1)
    k2 = np.expand_dims(rolling_windows2[idx_train],-1)
    k3 = np.expand_dims(rolling_windows3[idx_train],-1)
    # k4 = np.expand_dims(rolling_windows3[idx_train],-1)
    t_rolling = np.append(k1,k2,-1)
    t_rolling2 = np.append(t_rolling,k3,-1)
    # t_rolling3 = np.append(t_rolling2,k4,-1)
    self.train_set_vae = dict(data=t_rolling2)
    
    k1 = np.expand_dims(rolling_windows1[idx_val],-1)
    k2 = np.expand_dims(rolling_windows2[idx_val],-1)
    k3 = np.expand_dims(rolling_windows3[idx_val],-1)
    # k4 = np.expand_dims(rolling_windows4[idx_val],-1)
    t_rolling = np.append(k1,k2,-1)
    t_rolling2 = np.append(t_rolling,k3,-1)
    # t_rolling3 = np.append(t_rolling2,k4,-1)
    self.val_set_vae = dict(data=t_rolling2)

    k1 = np.expand_dims(rolling_windows1[idx_val[:self.config['batch_size']]],-1)
    k2 = np.expand_dims(rolling_windows2[idx_val[:self.config['batch_size']]],-1)
    k3 = np.expand_dims(rolling_windows3[idx_val[:self.config['batch_size']]],-1)
    # k4 = np.expand_dims(rolling_windows4[idx_val[:self.config['batch_size']]],-1)
    t_rolling = np.append(k1,k2,-1)
    t_rolling2 = np.append(t_rolling,k3,-1)
    # t_rolling3 = np.append(t_rolling2,k4,-1)
    self.test_set_vae = dict(data=t_rolling2)

    
    # self.train_set_vae = dict(data=np.expand_dims(rolling_windows[idx_train], -1))
    # self.val_set_vae = dict(data=np.expand_dims(rolling_windows[idx_val], -1))
    # self.test_set_vae = dict(data=np.expand_dims(rolling_windows[idx_val[:self.config['batch_size']]], -1))
    print(n_train_sample)
    #100 - 0 // 24 == 
    # create LSTM training and validation set
    for k in range(self.config['l_win']):
      n_not_overlap_wins = (n_train_sample - k) // self.config['l_win']
      n_train_lstm = n_not_overlap_wins - self.config['l_seq'] + 1
      print(n_train_lstm)
      cur_lstm_seq = np.zeros((n_train_lstm, self.config['l_seq'], self.config['l_win']))
      for i in range(n_train_lstm):
        cur_seq = np.zeros((self.config['l_seq'], self.config['l_win']))
        for j in range(self.config['l_seq']):
          # print(k,i,j)
          cur_seq[j] = data['training'][k + self.config['l_win'] * (j + i): k + self.config['l_win'] * (j + i + 1)]
        cur_lstm_seq[i] = cur_seq
      if k == 0:
        lstm_seq = cur_lstm_seq
      else:
        lstm_seq = np.concatenate((lstm_seq, cur_lstm_seq), axis=0)
    
    #CUSTOM create Lstm traing adn validation set
    for k in range(self.config['l_win']):
      n_not_overlap_wins = (n_train_sample - k) // self.config['l_win']
      n_train_lstm = n_not_overlap_wins - self.config['l_seq'] + 1
      cur_lstm_seq2 = np.zeros((n_train_lstm, self.config['l_seq'], self.config['l_win']))
      for i in range(n_train_lstm):
        cur_seq2 = np.zeros((self.config['l_seq'], self.config['l_win']))
        for j in range(self.config['l_seq']):
          # print(k,i,j)
          cur_seq2[j] = data2['training'][k + self.config['l_win'] * (j + i): k + self.config['l_win'] * (j + i + 1)]
        cur_lstm_seq2[i] = cur_seq2
      if k == 0:
        lstm_seq2 = cur_lstm_seq2
      else:
        lstm_seq2 = np.concatenate((lstm_seq2, cur_lstm_seq2), axis=0)
    
    for k in range(self.config['l_win']):
      n_not_overlap_wins = (n_train_sample - k) // self.config['l_win']
      n_train_lstm = n_not_overlap_wins - self.config['l_seq'] + 1
      cur_lstm_seq3 = np.zeros((n_train_lstm, self.config['l_seq'], self.config['l_win']))
      for i in range(n_train_lstm):
        cur_seq3 = np.zeros((self.config['l_seq'], self.config['l_win']))
        for j in range(self.config['l_seq']):
          # print(k,i,j)
          cur_seq3[j] = data3['training'][k + self.config['l_win'] * (j + i): k + self.config['l_win'] * (j + i + 1)]
        cur_lstm_seq3[i] = cur_seq3
      if k == 0:
        lstm_seq3 = cur_lstm_seq3
      else:
        lstm_seq3= np.concatenate((lstm_seq3, cur_lstm_seq3), axis=0)
    
    # for k in range(self.config['l_win']):
    #   n_not_overlap_wins = (n_train_sample - k) // self.config['l_win']
    #   n_train_lstm = n_not_overlap_wins - self.config['l_seq'] + 1
    #   cur_lstm_seq4 = np.zeros((n_train_lstm, self.config['l_seq'], self.config['l_win']))
    #   for i in range(n_train_lstm):
    #     cur_seq4 = np.zeros((self.config['l_seq'], self.config['l_win']))
    #     for j in range(self.config['l_seq']):
    #       # print(k,i,j)
    #       cur_seq4[j] = data4['training'][k + self.config['l_win'] * (j + i): k + self.config['l_win'] * (j + i + 1)]
    #     cur_lstm_seq4[i] = cur_seq4
    #   if k == 0:
    #     lstm_seq4 = cur_lstm_seq4
    #   else:
    #     lstm_seq4 = np.concatenate((lstm_seq4, cur_lstm_seq4), axis=0)


    # n_train_lstm_k = lstm_seq2
    idx_train, idx_val, self.n_train_lstm, self.n_val_lstm = self.separate_train_and_val_set(n_train_lstm)
    # print(lstm_seq[idx_train].shape)
    data2 = np.expand_dims(lstm_seq2[idx_train],-1)
    data = np.expand_dims(lstm_seq[idx_train],-1)
    data3 = np.expand_dims(lstm_seq3[idx_train],-1)
    # data4 = np.expand_dims(lstm_seq4[idx_train],-1)
    # data[-1].append(np.zeros())   
    k_test = np.append(data,data2,-1)
    print(k_test.shape)
    k_test2 = np.append(k_test,data3,-1)
    print(k_test2.shape)
    # k_test3 = np.append(k_test2,data4,-1)
    # print(k_test3.shape)

    data2_val = np.expand_dims(lstm_seq2[idx_val], -1)
    data_val = np.expand_dims(lstm_seq[idx_val], -1)
    data3_val = np.expand_dims(lstm_seq3[idx_val], -1)
    # data4_val = np.expand_dims(lstm_seq4[idx_val], -1)
    
    k_val = np.append(data_val,data2_val,-1)
    print(k_val.shape)
    k_val2 = np.append(k_val, data3_val,-1)
    print(k_val2.shape)
    # k_val3 = np.append(k_val2, data4_val, -1)
    # print(k_val3.shape)    
    data = dict(data=k_test2)
    data_val = dict(data = k_val2)
    
    self.train_set_lstm = data
    self.val_set_lstm = data_val
    print("train_set_lstm",self.train_set_lstm['data'].shape)
    # print("train_set_lstm",self.train_set_lstm['data'][0])

  # def plot_time_series(self, data, time, data_list):
  #   fig, axs = plt.subplots(1, 4, figsize=(18, 2.5), edgecolor='k')
  #   fig.subplots_adjust(hspace=.8, wspace=.4)
  #   axs = axs.ravel()
  #   for i in range(4):
  #     axs[i].plot(time / 60., data[:, i])
  #     axs[i].set_title(data_list[i])
  #     axs[i].set_xlabel('time (h)')
  #     axs[i].set_xlim((np.amin(time) / 60., np.amax(time) / 60.))
  #   savefig(self.config['result_dir'] + '/raw_training_set_normalised.pdf')
