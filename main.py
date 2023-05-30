import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from math import exp


plt.rcParams['axes.linewidth'] = 1  #
font = {'family': 'serif',
            'serif': 'Times New Roman',
            'weight': 'normal',
            'size': 14}
plt.rc('font', **font)

def sort_path(path):
    try:
        name = int((re.split('\+|,|_',os.path.basename(path))[2]))
    except:
        name = 0
    return name

def func_exp(x, param):
    return param[0] * np.exp(param[1] * x) + param[2]

def fitting_algorithm(data):
        for key in data.keys():
            # Удаление пустых строк (Nan):
            datas = data[key].dropna(axis=0, how='any')
            # Задание области данных для фиттирования [a1,b1]:
            a1 = 0.1
            b1 = 5  # 2
            dat = datas[(datas[f'End_time{high}']-datas[f'Begin_time{high}'] > a1) &
                        (datas[f'End_time{high}']-datas[f'Begin_time{high}'] < b1)]
            # Задание переменных x (ex) и y (ey) для фиттирования:
            ex = (datas[f'End_time{high}']-datas[f'Begin_time{high}'])
            ey = datas[f'Amplitude{high}']

            x = (dat[f'End_time{high}']-dat[f'Begin_time{high}'])
            y = dat[f'Amplitude{high}']
            # Фиттирование a*exp(b*x)+c:
            popt, pcov = curve_fit(lambda t, a, b, c: a * np.exp(b * t) + c, x, y, p0=(1.2, 0.0007, 10), maxfev=5000)
            # Ошибка для экспоненты:
            error_exp = np.sqrt(np.mean((func_exp(x, popt) - y) ** 2))
            print("Root Mean Squared Error 'exp' = ", error_exp)

            # # Фиттирование для полинома 3-ей степени:
            # z = np.polyfit(ex, ey, 3)
            # # Ошибка для полинома:
            # rmse = np.sqrt(np.mean((np.polyval(z, ex) - ey) ** 2))
            # print("Root Mean Squared Error 'poly' = ", rmse)
            # plt.scatter(ex, np.polyval(z, ex), label="fit 'poly': a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f" % tuple(z))

            plt.scatter(ex, func_exp(ex, popt), label="fit 'exp': a=%5.3f, b=%5.3f, c=%5.3f" % tuple(popt))
            plt.scatter(ex, func_exp(ex - 0.5, popt) - 5, label='Смещение') # - 0.2 по x
            plt.scatter(ex, ey, label='data no scintillator')
            plt.xlabel('Ширина, мкс', fontsize=14)
            plt.ylabel('А, коды АЦП', fontsize=14)
            plt.title('Распределение амплитуд от ширины'.format(h=high), fontweight='normal')
            plt.legend()
            plt.show()

            # # Фиттирование для полинома 3-ей степени:
            # z = np.polyfit(ex, ey, 3)
            # # Ошибка для полинома:
            # rmse = np.sqrt(np.mean((np.polyval(z, ex) - ey) ** 2))
            # print("Root Mean Squared Error 'poly' = ", rmse)
            # plt.scatter(ex, np.polyval(z, ex), label="fit 'poly': a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f" % tuple(z))

        return {'exp': popt}

def sort_pulses(datas_on_range, param):
    sorted_pulses = {}
    for key in datas_on_range.keys():
        table = datas_on_range[key]
        sorted_pulses[key] = table[(table[f'Amplitude{high}']>=6) &
                                   ((table[f'End_time{high}']-table[f'Begin_time{high}']) >= 4) &
                                   ((table[f'End_time{high}']-table[f'Begin_time{high}'])<9)]

        # delta = list(table[f'End_time{high}']-table[f'Begin_time{high}'])
        # mask = list(map(lambda x: param[0]*exp(param[1]*(x-0.075)) + param[2] - 4, delta))
        # sorted_pulses[key] = table[(table[f'Amplitude{high}'] < mask) & (table[f'Amplitude{high}'] != 0)]
    return sorted_pulses

def pulses_not_n(data, param):
    noises = {}
    for key in data.keys():
        array = data[key]
        # sorted_pulses[key] = array[(array[f'Amplitude{high}']>=6) & (array[f'Amplitude{high}']<60) &
        #                            ((array[f'End_time{high}']-array[f'Begin_time{high}']) >= 3) &
        #                            ((array[f'End_time{high}']-array[f'Begin_time{high}'])<7)]

        delta = list(array[f'End_time{high}']-array[f'Begin_time{high}'])
        mask = list(map(lambda x: param[0]*exp(param[1]*(x-0.5)) + param[2] - 1, delta))
        noises[key] = array[(array[f'Amplitude{high}'] >= mask)]
    return noises

def counts_per_seconds(sorted):
    counts_per_second = {}
    for key in sorted.keys():
        table = sorted[key]
        counts_per_second[key] = (float(len(table))/1000) #/100 для 5000 тыс
    return counts_per_second

def counts_of_neutron(Data):
    return 0

high = ''
Scintillator = ''
if __name__ == '__main__':
    # os.chdir("C:/Users/funny/OneDrive/Рабочий стол/Учеба МИФИ/НИРС/3семестр/Разработка алгоритма/на расстояниях/ФЭУ-200_LRB2_A4_на расстояниях/Данные")
    # os.chdir("C:/Users/funny/OneDrive/Рабочий стол/Учеба МИФИ/НИРС/3семестр/Разработка алгоритма/Файлы с данными 50000/EMI/EMI_noscint/Данные")
    # os.chdir("C:/Users/funny/OneDrive/Рабочий стол/Учеба МИФИ/НИРС/3семестр/Разработка алгоритма/аппробация алгоритма на УРАНе/Данные")
    # os.chdir("C:/Users/funny/OneDrive/Рабочий стол/Учеба МИФИ/НИРС/3семестр/Разработка алгоритма/EMI новый делитель/Данные")
    os.chdir("C:/Users/funny/OneDrive/Рабочий стол/Учеба МИФИ/НИРС/3семестр/Разработка алгоритма/Рабочие файлы ФЭУ-200/ФЭУ-200_noscint/Данные")

    paths = [path for path in Path.cwd().rglob("*.txt")]
    paths = sorted(paths, key=sort_path)
    print(paths)
    datas_on_range = {}
    datas_on_range_nosource = {}
    datas_noscint = {}
    try:
        os.chdir(f"C:/Users/funny/OneDrive/Рабочий стол/Учеба МИФИ/НИРС/3семестр/Разработка алгоритма/Рабочие файлы ФЭУ-200/ФЭУ-200_noscint/TEST_{high}")
    except:
        os.mkdir(f"C:/Users/funny/OneDrive/Рабочий стол/Учеба МИФИ/НИРС/3семестр/Разработка алгоритма/Рабочие файлы ФЭУ-200/ФЭУ-200_noscint/TEST_{high}")
        os.chdir(f"C:/Users/funny/OneDrive/Рабочий стол/Учеба МИФИ/НИРС/3семестр/Разработка алгоритма/Рабочие файлы ФЭУ-200/ФЭУ-200_noscint/TEST_{high}")

    for path in paths:
        if re.split('\.|,|_',os.path.basename(path))[1] == 'source':
            Data = pd.read_table('{}'.format(path),sep='\s+', engine='python')
            datas_on_range[(re.split('\.|,|_', os.path.basename(path))[0])] = Data
        elif re.split('\.|,|_',os.path.basename(path))[1] == 'nosource':
            Data = pd.read_table('{}'.format(path), sep='\s+', engine='python')
            datas_on_range_nosource[(re.split('\.|,|_', os.path.basename(path))[0])] = Data
        else:
            Data = pd.read_table('{}'.format(path), sep='\s+', engine='python')
            datas_noscint[(re.split('\.|,|_', os.path.basename(path))[0])] = Data
    print('File has been read')

    fitting = fitting_algorithm(datas_on_range)
    try:
        param = fitting['exp']
    except:
        param = 0

    sorted = sort_pulses(datas_on_range, param)
    counts_per_second = counts_per_seconds(sorted)
    for key in datas_on_range_nosource.keys():
        data = datas_on_range_nosource[key]
        datas_on_range_nosource[key] = data[(data[f'End_time{high}'] - data[f'Begin_time{high}']) != 0]
    sorted_nosource = sort_pulses(datas_on_range_nosource, param)
    counts_per_second_nosource = counts_per_seconds(sorted_nosource)
    imp_not_n_source = pulses_not_n(datas_on_range, param)
    counts_per_second_noise_source = counts_per_seconds(imp_not_n_source)
    imp_not_n_nosource = pulses_not_n(datas_on_range_nosource, param)
    counts_per_second_noise_nosource = counts_per_seconds(imp_not_n_nosource)

    # Построение графиков запаздывания нейтронных импульсов
    # for keys in sorted.keys():
    #     filter_time = [_ for _ in sorted[keys]['Pulse_time'] if float(_) > 150]
    #     counts, bins = np.histogram(datas_on_range[keys]['Pulse_time'], bins=range(0, 19999, 10))
    #     norm = sum(counts)
    #     plt.hist(bins[:-1], bins=range(150, 19999, 500), weights=counts/norm, label=f'{keys}')
    #     plt.legend()
    #     plt.ylim(0, 0.2)
    #     # plt.xlim(-10,199)
    #     plt.title('{k}'.format(k=keys), fontweight='normal')
    #     plt.savefig('{k} График запаздывания нейтронных импульсов.png'.format(k=keys))
    #     plt.close()
  
    with open(f'Темп счета{high}.txt', mode='w') as file:
        file.write('№' + '\t' + 'Без источника' + '\t' + 'Источник' + '\t' + 'Шумы c источником' +
                   '\t' + 'Шумы без источника' + '\n')
        for key in counts_per_second:
            try:
                file.write(f'{key}' + '\t' + str(counts_per_second_nosource[f'{key}']) + '\t' +
                           str(counts_per_second[f'{key}']) + '\t' + str(counts_per_second_noise_source[f'{key}']) +
                           '\t' + str(counts_per_second_noise_nosource[f'{key}']) + '\n')
            except:
                file.write(f'{key}' + '\t' + 'no data' + '\t' + str(counts_per_second[f'{key}']) + '\n')

# Распределение ширины от амплитуды (вывод одновременный УРАН 12 каналов)
#     i = 0
#     j = 0
#     C = 0
#     plt.title('Распределение амплитуды от ширины')
#     for keys in datas_on_range:
#             try:
#                 plt.subplot2grid((4,3), (i, j))
#                 plt.scatter(datas_on_range[keys][f'End_time{high}'] - datas_on_range[keys][f'Begin_time{high}'],
#                             datas_on_range[keys][f'Amplitude{high}'], s=0.5, label='{k}'.format(k=keys))
#             except:
#                 print('no file "source"')
#
#             if (C+1)%4 == 0:
#                 i = 0
#                 j += 1
#             else:
#                 i += 1
#             C += 1
#
#     for keys in datas_on_range_nosource:
#             try:
#                 DATA = datas_on_range_nosource[keys]
#                 plt.subplot2grid((4, 3), (i, j))
#                 plt.scatter(DATA[f'End_time{high}'] - DATA[f'Begin_time{high}'],
#                                 DATA[f'Amplitude{high}'], s=0.5, label='{}_nosource'.format(keys))
#             except:
#                 print('no file "nosource"')
#
#             if (C+1)%4 == 0:
#                 i = 0
#                 j += 1
#             else:
#                 i += 1
#             C += 1
#
#             plt.ylim(0, 100)
#             plt.xlim(0, 100)
#             plt.xticks(range(7))
#             plt.yticks(range(0, 100, 25))
#             plt.legend(fontsize=7)
#         # plt.subplots_adjust(hspace=0, wspace=0) -----
#     plt.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.5)
#     plt.show()
#     plt.close()
    print()

# Распределение ширины от амплитуды
    for keys in datas_on_range:
        try:
            plt.scatter(datas_on_range[keys][f'End_time{high}'] - datas_on_range[keys][f'Begin_time{high}'],
                        datas_on_range[keys][f'Amplitude{high}'], s=0.5, label='{}'.format(keys))
            # plt.scatter(datas_on_range[keys][f'End_time9'] - datas_on_range[keys][f'Begin_time9'],
            #             datas_on_range[keys][f'Amplitude9'], s=0.5, label='{} 9'.format(keys))
            # plt.ylim(0, 50)
            # plt.xlim(0, 7)
            # plt.xticks(range(7))
            # plt.yticks(range(0, 50, 10))
            # plt.xlabel('Ширина, мкс', fontsize=14)
            # plt.ylabel('А, коды АЦП', fontsize=14)
            # plt.title(f'Распределение амплитуд от ширины', fontweight='normal')
            # plt.legend()
            # plt.savefig(f'{keys}.png', dpi=1000)
            # plt.close()
        except:
            continue

            #Отобразить отсортированные
            # plt.scatter(sorted[keys][f'End_time{high}'] - sorted[keys][f'Begin_time{high}'],
            #             sorted[key][f'Amplitude{high}'], s=1, label='sorted')

    for keys in datas_on_range_nosource.keys():
        try:
            DATA = datas_on_range_nosource[keys]
            plt.scatter(DATA[f'End_time{high}'] - DATA[f'Begin_time{high}'],
                            DATA[f'Amplitude{high}'], s=2.5, label='{}_nosource'.format(keys))
            # plt.ylim(0, 500)
            # plt.xlim(0, 12)
            # plt.xticks(range(12))
            # plt.yticks(range(0, 500, 25))
            # plt.xlabel('Ширина, мкс', fontsize=14)
            # plt.ylabel('А, коды АЦП', fontsize=14)
            # plt.title('Распределение амплитуд от ширины'.format(h=high), fontweight='normal')
            # plt.legend()
            # plt.savefig('{k} Распределение амплитуд от ширины с источником.png'.format(h=high, k=keys), dpi=1000)
            # plt.show()
            # plt.close()
        except:
            continue

    for key in datas_noscint.keys():
            x_noscint = datas_noscint[key][f'End_time{high}'] - datas_noscint[key][f'Begin_time{high}']
            y_noscint = datas_noscint[key][f'Amplitude{high}']

            param_exp = fitting['exp']

            # try:
            #     x = datas_on_range[keys][f'End_time{high}'] - datas_on_range[keys][f'Begin_time{high}']
            #     plt.scatter(x, func_exp(x - 0.5, param_exp) - 1, s=1,
            #                 label="fit 'exp': a=%5.3f, b=%5.3f, c=%5.3f" % tuple(param_exp))
            #
            # except:
            #     continue

    #
    #         # noscintillator
    #         plt.scatter(x_noscint, y_noscint, label='no scintillator', s = 1)
    #         print('Parameter for "exp":', param_exp)

    plt.ylim(0, 250)
    plt.xlim(0, 12)
    plt.xticks(range(0, 12, 1))
    plt.yticks(range(0, 250, 25))
    plt.xlabel('Ширина, мкс', fontsize=14)
    plt.ylabel('А, коды АЦП', fontsize=14)
    plt.title('Распределение амплитуд от ширины'.format(h=high), fontweight='normal')
    plt.legend()
    plt.savefig('{h} Распределение амплитуд от ширины с источником.png'.format(h=high), )
    plt.show()
    plt.close()

#Распределение амплитуд
    for key in datas_on_range:
        DATA_source = datas_on_range[key]
        # print(DATA_source[f'Amplitude{high}'])
        counts, bins = np.histogram(DATA_source[f'Amplitude{high}'], bins=range(6, 300, 1))
        plt.hist(bins[:-1], bins=range(6, 300, 1), weights=counts, align='left', label=f'Источник {key}')
        try:
            DATA_nosource = datas_on_range_nosource[key]
            counts1, bins1 = np.histogram(DATA_nosource[f'Amplitude{high}'], bins=range(6, 250, 1))
            plt.hist(bins1[:-1], bins=range(6, 300, 1), weights=counts1, align='left', alpha=0.5,
                      label=f'Без источника {key}')
        except:
            pass

        plt.minorticks_on()
        # включаем основную сетку
        plt.grid(which='major')
        # включаем дополнительную сетку
        plt.grid(which='minor', linestyle=':')
        plt.tight_layout()

        plt.title(f'Распределение амплитуд{high}', fontweight='normal')
        plt.xlabel('log(A[коды АЦП])', fontsize=14)
        plt.ylabel('log(N)', fontsize=14)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.savefig('Распределение амплитуд {h}{key}.png'.format(h=high, key=key), bbox_inches = 'tight')
        plt.close()
        # plt.show()

#Распределение ширин
    for key in datas_on_range:
        DATA_source = datas_on_range[key]

        counts, bins = np.histogram(DATA_source[f'End_time{high}']-DATA_source[f'Begin_time{high}'], bins=range(0, 11, 1))
        plt.hist(bins[:-1], bins=range(0, 10, 1), weights=counts, label=f'{key}')

        try:
            DATA_nosource = datas_on_range_nosource[key]
            counts1, bins1 = np.histogram(DATA_nosource[f'End_time{high}']-DATA_nosource[f'Begin_time{high}'], bins=range(0, 11, 1))
            plt.hist(bins1[:-1], bins=range(0, 10, 1), weights=counts1, alpha=0.5, label=f'Без источника {key}')
        except:
            pass
        plt.xticks(np.linspace(0, 10, 11))
        plt.minorticks_on()
        # включаем основную сетку
        plt.grid(which='major')
        # включаем дополнительную сетку
        plt.grid(which='minor', linestyle=':')
        plt.tight_layout()
        plt.xlabel('Ширина, мкс', fontsize=14)
        plt.ylabel('Число импульсов', fontsize=14)
        # plt.title(f'Распределение ширин{high}', fontweight='normal')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.ylim(0, 2500)
        plt.legend()
        plt.savefig('Распределение ширин {h}{key}.png'.format(h=high, key=key), bbox_inches = 'tight')
        plt.close()
        # plt.show()

# Распределение амплитуд с условиями
    for key in sorted:
            DATA_sort = sorted[key]
            try:
                DATAno_sort = sorted_nosource[key]
                DATA_source = datas_on_range[key]
            except:
                pass

            # plt.scatter((DATA_sort[f'End_time{high}']-DATA_sort[f'Begin_time{high}']),DATA_sort[f'Amplitude{high}'])
            # plt.show()

            counts, bins = np.histogram(DATA_sort[f'Amplitude{high}'], bins=range(6, 400, 1))
            plt.hist(bins[:-1], bins=range(6, 400, 1), weights=counts, color='black', align='left', alpha=0.7,
                     label=f'С отбором {key}')
            plt.xscale('log')
            plt.yscale('log')

            # counts1, bins1 = np.histogram(DATAno_sort[f'Amplitude{high}'], bins=range(6, 400, 1))
            # plt.hist(bins1[:-1], bins=range(6, 400, 1), weights=counts1, color='red', align='left', alpha=0.5,
            #             label=f'С отбором Без источника {key}')

            counts2, bins2 = np.histogram(DATA_source[f'Amplitude{high}'], bins=range(6, 400, 1))

            plt.hist(bins2[:-1], bins=range(6, 400, 1), weights=counts2, align='left', alpha=0.5,
                     label=f'Источник {key}')

            # counts3 = np.histogram(DATA_sort[f'Amplitude{high}'], bins=range(6, 400, 1))[0] - np.histogram(DATAno_sort[f'Amplitude{high}'], bins=range(6, 400, 1))[0]
            # print(np.histogram(DATA_sort[f'Amplitude{high}'], bins=range(6, 400, 1))[0])
            # print(np.histogram(DATAno_sort[f'Amplitude{high}'], bins=range(6, 400, 1))[0])
            # print(counts3)
            # plt.hist(bins[:-1] ,bins=range(6, 400, 1), weights=counts3, align='left', alpha=0.3,
            #          label=f'С отбором (разность) {key}')

            # counts3 = np.histogram(DATA_source[f'Amplitude{high}'], bins=range(6, 200, 1))[0] - np.histogram(DATA_sort[f'Amplitude{high}'], bins=range(6, 200, 1))[0]
            # plt.hist(bins2[:-1], bins=range(6, 200, 1), color='black', weights=counts3, align='left', alpha=0.3,
            #          label=f'resid {key}')

            plt.minorticks_on()
            # включаем основную сетку
            plt.grid(which='major')
            # включаем дополнительную сетку
            plt.grid(which='minor', linestyle=':')
            plt.tight_layout()

            plt.title(f'Распределение амплитуд{high} отбор', fontweight='normal')
            plt.xlabel('log(A), коды АЦП', fontsize=14)
            plt.ylabel('log(N)', fontsize=14)
            plt.xscale('log')
            plt.yscale('log')
            plt.ylim(0.7, 1000000)
            plt.legend()
            plt.savefig('Распределение амплитуд {h}{key} отбор.png'.format(h=high, key=key), bbox_inches = 'tight')
            # plt.show()
            plt.close()

# Распределение ширин с условиями
    for key in sorted:

        DATA_source = sorted[key]
        counts, bins = np.histogram(DATA_source[f'End_time{high}'] - DATA_source[f'Begin_time{high}'],
                                        bins=range(0, 11, 1))
        plt.hist(bins[:-1], bins=range(0, 10, 1), weights=counts, label=f'С источником {key}')
        #

        try:
            DATA_nosource = sorted_nosource[key]
            counts1, bins1 = np.histogram(DATA_nosource[f'End_time{high}'] - DATA_nosource[f'Begin_time{high}'],
                                              bins=range(0, 11, 1))
            plt.hist(bins1[:-1], bins=range(0, 10, 1), weights=counts1, alpha=0.5, label=f'Без источника {key}')
        except:
            pass

        plt.xticks(np.linspace(0, 10, 11))
        plt.minorticks_on()
        # включаем основную сетку
        plt.grid(which='major')
        # включаем дополнительную сетку
        plt.grid(which='minor', linestyle=':')
        plt.tight_layout()

        plt.title(f'Распределение ширин{high} отбор', fontweight='normal')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.legend()
        plt.savefig('Распределение ширин {h}{key} отбор.png'.format(h=high, key=key), bbox_inches = 'tight')
        # plt.show()
        plt.close()

#Темп счета
    keys_x = [re.split('\+|,|_',os.path.basename(k))[2] for k in counts_per_second.keys()]
    plt.scatter(keys_x, counts_per_second.values(), color='black', label='Все импульсы')

    try:
        plt.scatter(counts_per_second_nosource.keys(),
                    pd.DataFrame(counts_per_second.values())-pd.DataFrame(counts_per_second_nosource.values()),
                    color='red', label='С вычетом без источника')
    except:
        pass

    # plt.ylim(0, 1.5)
    plt.xlabel('Расстояние, см', fontsize=14)
    plt.ylabel('Темп счета, с^-1', fontsize=14)
    plt.title('Темп счета ({h})'.format(h=high), fontweight='normal')
    plt.legend()
    plt.savefig('Темп счета {h}.png'.format(h=high), bbox_inches = 'tight')
    # plt.show()
    plt.close()
