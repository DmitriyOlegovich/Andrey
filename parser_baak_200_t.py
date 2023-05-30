import struct

#try
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import re
import time

import pandas as pd
from tqdm import tqdm
from decorators.time_finc import time_func

#Задание шрифта
plt.rcParams['axes.linewidth'] = 1  #
font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'normal',
        'size': 14}
plt.rc('font', **font)
#end_of_try

MEMORY_BUFFER_SIZE = 150000  # bytes
FAKE_HEADER_LENGTH = 24  # bytes
HEADER_LENGTH = 40  # bytes
FOOTER_LENGTH = 4  # bytes
NUMBER_OF_CHANNELS = 12
LENGTH_OF_CHANNEL_DATA_STD = 2048  # bytes
LENGTH_OF_CHANNEL_DATA_TAIL = 40000  # bytes
EXPECTED_DATA_PACKAGE_LENGTH = FAKE_HEADER_LENGTH + NUMBER_OF_CHANNELS * LENGTH_OF_CHANNEL_DATA_TAIL +\
                               HEADER_LENGTH + NUMBER_OF_CHANNELS * LENGTH_OF_CHANNEL_DATA_STD + FOOTER_LENGTH  # bytes

#try

ALL_list_of_a = [list() for _ in range(12)]
ALL_list_of_t_start = [list() for _ in range(12)]
ALL_list_of_t_end = [list() for _ in range(12)]
ALL_list_of_t5_start = [list() for _ in range(12)]
ALL_list_of_t5_end = [list() for _ in range(12)]
ALL_list_of_t7_start = [list() for _ in range(12)]
ALL_list_of_t7_end = [list() for _ in range(12)]
ALL_list_of_t9_start = [list() for _ in range(12)]
ALL_list_of_t9_end = [list() for _ in range(12)]
ALL_list_of_amplitude = [list() for _ in range(12)]
ALL_list_of_amplitude5 = [list() for _ in range(12)]
ALL_list_of_amplitude7 = [list() for _ in range(12)]
ALL_list_of_amplitude9 = [list() for _ in range(12)]
ALL_list_of_event_times = [list() for _ in range(12)]


#end_of_try

FOOTER = b'\xee\xee\xee\xee'
CORRECT_PACKAGE_END = b'\xff\xff\xff\xff'
DISTORTED_PACKAGE_END = b'\xfe\xfe\xfe\xfe'


def parse_time(data_0, data_1, data_2, data_3):
    dns = data_0 & 0x7f
    mks = (data_0 & 0xff80) >> 7
    mks |= (data_1 & 1) << 9
    mls = (data_1 & 0x7fe) >> 1
    s = (data_1 & 0xf800) >> 11
    s |= (data_2 & 1) << 5
    m = (data_2 & 0x7e) >> 1
    h = (data_2 & 0xf80) >> 7
    return {"hours": int(h), "minutes": int(m), "seconds": int(s),
            "milliseconds": int(mls), "microseconds": int(mks), "nanoseconds": int(dns) * 10,
            "text": "{:02}:{:02}:{:02}.{:03}.{:03}.{:03}".format(
                int(h), int(m), int(s), int(mls), int(mks), int(dns)*10)}


def parse_time_with_days(event_time_pack):
    dns = event_time_pack[0] & 0x7f
    mks = ((event_time_pack[2] & 0x1) << 9) | (event_time_pack[1] << 1) | (event_time_pack[0] >> 7)
    mls = ((event_time_pack[3] & 0x07) << 7) | ((event_time_pack[2] >> 1) & 0x7f)
    s = ((event_time_pack[4] & 0x01) << 5) | (event_time_pack[3]) >> 3
    m = (event_time_pack[4] >> 1) & 0x3f
    h = ((event_time_pack[5] & 0x0f) << 1) | (event_time_pack[4] >> 7)
    d = ((event_time_pack[6] & 0x03) << 4) | (event_time_pack[5] >> 4)
    return {"days": int(d), "hours": int(h), "minutes": int(m), "seconds": int(s),
            "milliseconds": int(mls), "microseconds": int(mks), "nanoseconds": int(dns) * 10,
            "text": "{:02}_{:02}:{:02}:{:02}.{:03}.{:03}.{:03}".format(
                int(d), int(h), int(m), int(s), int(mls), int(mks), int(dns)*10)}


def split_data_bin_to_packages(data_bin):
    return data_bin.split(CORRECT_PACKAGE_END)


def check_and_remove_distorted_packages(data_packages):
    number_of_distorted_data_packages = 0
    for idx, data_package in enumerate(data_packages):
        if DISTORTED_PACKAGE_END in data_package:
            number_of_distorted_data_packages += 1
            data_packages[idx] = data_package.split(DISTORTED_PACKAGE_END)[-1]
    return data_packages, number_of_distorted_data_packages


def parse_package(data_package):
    fake_header = data_package[0:24]
    channels_data_tail = data_package[24:(480000 + 24)]
    header = data_package[(480000 + 24):(480000 + 24 + 40)]
    channels_data_std = data_package[(480000 + 24 + 40):]
    # number_of_requests = struct.unpack("I", header[0:4])[0]
    # number_of_triggers = struct.unpack("I", header[4:8])[0]
    event_time = parse_time(struct.unpack("H", header[8:10])[0],
                            struct.unpack("H", header[10:12])[0],
                            struct.unpack("H", header[12:14])[0],
                            struct.unpack("H", header[14:16])[0])["text"]
    # event_time_with_days = parse_time_with_days(header[8:16])["text"]

    wf_t = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: []}
    len_of_ch_data = LENGTH_OF_CHANNEL_DATA_TAIL
    for ch_num in range(NUMBER_OF_CHANNELS):
        for i in range(int(LENGTH_OF_CHANNEL_DATA_TAIL / 2)):
            value = struct.unpack(
                "H", channels_data_tail[(ch_num * len_of_ch_data + 2 * i):(ch_num * len_of_ch_data + 2 * i + 2)])[0]
            wf_t[value >> 12].append(value & 0xFFF)

    wf_s = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: []}
    len_of_ch_data = LENGTH_OF_CHANNEL_DATA_STD
    for ch_num in range(NUMBER_OF_CHANNELS):
        for i in range(int(LENGTH_OF_CHANNEL_DATA_STD/2)):
            value = struct.unpack(
                "H", channels_data_std[(ch_num * len_of_ch_data + 2 * i):(ch_num * len_of_ch_data + 2 * i + 2)])[0]
            wf_s[value >> 12].append(value & 0xFFF)
    return {"ts": event_time,
            "wf_s": {"ch_00": wf_s[0], "ch_01": wf_s[1], "ch_02": wf_s[2], "ch_03": wf_s[3],
                     "ch_04": wf_s[4], "ch_05": wf_s[5], "ch_06": wf_s[6], "ch_07": wf_s[7],
                     "ch_08": wf_s[8], "ch_09": wf_s[9], "ch_10": wf_s[10], "ch_11": wf_s[11]},
            "wf_t": {"ch_00": wf_t[0], "ch_01": wf_t[1], "ch_02": wf_t[2], "ch_03": wf_t[3],
                     "ch_04": wf_t[4], "ch_05": wf_t[5], "ch_06": wf_t[6], "ch_07": wf_t[7],
                     "ch_08": wf_t[8], "ch_09": wf_t[9], "ch_10": wf_t[10], "ch_11": wf_t[11]}}


def parse_binary_data(data_bin):
    data_packages, number_of_distorted_packages =\
        check_and_remove_distorted_packages(split_data_bin_to_packages(data_bin))
    parsed_packages = [parse_package(data_package) for data_package in data_packages if len(data_package) == EXPECTED_DATA_PACKAGE_LENGTH]
    first_package_bin = data_packages[0] if len(data_packages[0]) != EXPECTED_DATA_PACKAGE_LENGTH else b''
    last_package_bin = data_packages[-1] if len(data_packages[-1]) != EXPECTED_DATA_PACKAGE_LENGTH else b''
    return parsed_packages, number_of_distorted_packages, first_package_bin, last_package_bin

#try!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
def take_baseline(waveform):
    sum_for_baseline = 0
    # plt.plot(waveform)
    # plt.show()
    for interval in range(150):
        try:
            sum_for_baseline += waveform[interval]
            baseline = sum_for_baseline / 150.0
        except:
            baseline = 2040
    return baseline

def save_file(folder,list_of_event_times, list_of_amplitude, list_of_amplitude5, list_of_amplitude7, list_of_amplitude9,
              list_of_a, list_of_t_start, list_of_t5_start, list_of_t7_start, list_of_t9_start,
              list_of_t_end, list_of_t5_end, list_of_t7_end, list_of_t9_end):
    if not os.path.isdir("{}".format(folder)):
        os.mkdir("{}".format(folder))
    for i in range(12):
        file_amplitude = open('{folder}/{folder}_{number}.txt'.format(folder=folder, number = i), 'w')
        file_amplitude.write(
            '№' + '\t' + 'Event_time' + '\t' + 'Amplitude' + '\t' + 'Begin_time' + '\t' + 'End_time' +
            '\t' + 'Amplitude5' + '\t' + 'Begin_time_on_5' + '\t' + 'End_time_on_5' +
            '\t' + 'Amplitude7' + '\t' + 'Begin_time_on_7' + '\t' + 'End_time_on_7' +
            '\t'+ 'Amplitude9' + '\t' + 'Begin_time_on_9' + '\t' + 'End_time_on_9' + '\n')
        file_amplitude.close()

    for i in range(12):
        file_amplitude = open('{folder}/{folder}_{number}.txt'.format(folder=folder, number = i), 'a')
        number = 0
        for element in list_of_amplitude[i]:
            number += 1
            file_amplitude.write(str(number) +'\t'+ str(list_of_event_times[i][number-1])+'\t'+
                                 str(list_of_amplitude[i][number-1])+
                                 '\t'+str(list_of_t_start[i][number-1])+'\t'+str(list_of_t_end[i][number-1]) +'\t'
                                 +str(list_of_amplitude5[i][number-1])+'\t'
                                 +str(list_of_t5_start[i][number-1])+'\t'+str(list_of_t5_end[i][number-1])+'\t'+
                                 str(list_of_amplitude7[i][number-1])+'\t'+str(list_of_t7_start[i][number-1])+'\t'+
                                 str(list_of_t7_end[i][number-1])+ '\t' +str(list_of_amplitude9[i][number-1])+'\t'
                                 +str(list_of_t9_start[i][number-1])+'\t'+str(list_of_t9_end[i][number-1])+'\n')

        file_amplitude.close()

    for i in range(12):
            file_amplitude = open('{folder}/Amplitude{number}.txt'.format(folder=folder, number=i), 'a')
            number = 0
            for element in list_of_a[i]:
                number += 1
                file_amplitude.write(str(number) + '\t' + str(element)+'\n')
            file_amplitude.close()

    return "file has been saved"

# @time_func
def take_number_of_neutron(waveform_tail, baseline, channel_number, package, time_of_pack):
    number_of_neutron = 0
    time = 0

    while time < 20000:
        if waveform_tail[time] - baseline >= 3 and time < 19999:
            maximum_amplitude = 0
            maximum_amplitude5 = 0
            maximum_amplitude7 = 0
            maximum_amplitude9 = 0
            neutron_time = 0
            time_found = time
            time_5 = time
            time_7 =time
            time_9 = time
            end = 0.0
            end5 = 0.0
            end7 = 0.0
            end9 = 0.0
            begin = 0.0
            begin5 = 0.0
            begin7 = 0.0
            begin9 = 0.0

            while (waveform_tail[time_found] - baseline) >= 3 and time_found < 19999 and time_found > 0:
                begin = time_found
                time_found -= 1
                if (waveform_tail[time_found] - baseline) < 3 and (waveform_tail[time_found] - baseline) != 3:
                    delta_wave = waveform_tail[time_found+1]-waveform_tail[time_found]
                    begin = (3.0 + baseline - (waveform_tail[time_found] - delta_wave*time_found))/delta_wave

            time_found = time

            while (waveform_tail[time_found] - baseline) >= 3 and time_found < 19999:
                if maximum_amplitude < (waveform_tail[time_found] - baseline):
                    maximum_amplitude = (waveform_tail[time_found] - baseline)
                    neutron_time = time_found
                end = time_found
                time_found += 1
                if (waveform_tail[time_found] - baseline) < 3 and (waveform_tail[time_found] - baseline) != 3:
                    delta_wave = waveform_tail[time_found] - waveform_tail[time_found - 1]
                    end = (3 + baseline - (waveform_tail[time_found] - delta_wave*time_found))/delta_wave

            while (waveform_tail[time_5] - baseline) >= 5 and time_5 < 19999 and time_5 > 0:
                begin5 = time_5
                time_5 -= 1
                if (waveform_tail[time_5] - baseline) < 5 and (waveform_tail[time_5] - baseline) != 5:
                    delta_wave = waveform_tail[time_5 + 1] - waveform_tail[time_5]
                    begin5 = (5.0 + baseline - (waveform_tail[time_5] - delta_wave * time_5)) / delta_wave

            time_5 = time

            while (waveform_tail[time_5] - baseline) >= 5 and time_5 < 19999 and time_5 > 0:
                if maximum_amplitude5 < (waveform_tail[time_5] - baseline):
                    maximum_amplitude5 = (waveform_tail[time_5] - baseline)
                end5 = time_5

                if (waveform_tail[time_5] - baseline) >= 7 and time_7 == 0:
                    time_7 = time_5

                time_5 += 1

                if (waveform_tail[time_5] - baseline) < 5 and (waveform_tail[time_5] - baseline) != 5:
                    delta_wave = waveform_tail[time_5] - waveform_tail[time_5 - 1]
                    end5 = (5.0 + baseline - (waveform_tail[time_5] - delta_wave * time_5)) / delta_wave

            time_77 = time_7

            while (waveform_tail[time_7] - baseline) >= 7 and time_7 < 19999 and time_7 > 0:
                time_7 -= 1
                if (waveform_tail[time_7] - baseline) <= 7:
                    try:
                        delta_wave = waveform_tail[time_7 + 1] - waveform_tail[time_7]
                        begin7 = (7.0 + baseline - (waveform_tail[time_7] - delta_wave * time_7)) / delta_wave
                    except:

                        begin7 = time_7
                        continue

            time_7 = time_77

            while (waveform_tail[time_7] - baseline) >= 7 and time_7 < 19999 and time_7 > 0:

                if maximum_amplitude7 < (waveform_tail[time_7] - baseline):
                    maximum_amplitude7 = (waveform_tail[time_7] - baseline)

                end7 = time_7

                if (waveform_tail[time_7] - baseline)>=9 and time_9 == 0:
                    time_9 = time_7
                time_7 += 1

                if (waveform_tail[time_7] - baseline) <= 7:
                    try:
                        delta_wave = waveform_tail[time_7] - waveform_tail[time_7 - 1]
                        end7 = (7.0 + baseline - (waveform_tail[time_7] - delta_wave * time_7)) / delta_wave
                    except:
                        end7 = time_7
                        continue

            time_99 = time_9

            while (waveform_tail[time_9] - baseline) >= 9 and time_9 < 19999 and time_9 > 0:

                begin9 = time_9
                time_9 -= 1
                if (waveform_tail[time_9] - baseline) <= 9:
                    try:
                        delta_wave = waveform_tail[time_9 + 1] - waveform_tail[time_9]
                        begin9 = (9.0 + baseline - (waveform_tail[time_9] - delta_wave * time_9)) / delta_wave
                    except:
                        begin9 = time_9
                        continue

            time_9 = time_99

            while (waveform_tail[time_9] - baseline) >= 9 and time_9 < 19999 and time_9 > 0:
                if maximum_amplitude9 < (waveform_tail[time_9] - baseline):
                    maximum_amplitude9 = (waveform_tail[time_9] - baseline)
                end9 = time_9
                time_9 += 1
                if (waveform_tail[time_9] - baseline) <= 9:
                    try:
                        delta_wave = waveform_tail[time_9] - waveform_tail[time_9 - 1]
                        end9 = (9.0 + baseline - (waveform_tail[time_9] - delta_wave * time_9)) / delta_wave
                    except:
                        end9 = time_9

            list_of_t_start[channel_number].append(begin)
            list_of_t_end[channel_number].append(end)
            list_of_amplitude[channel_number].append(maximum_amplitude)
            list_of_amplitude7[channel_number].append(maximum_amplitude7)
            list_of_amplitude5[channel_number].append(maximum_amplitude5)
            list_of_amplitude9[channel_number].append(maximum_amplitude9)
            list_of_event_times[channel_number].append(time_of_pack)
            list_of_t7_start[channel_number].append(begin7)
            list_of_t7_end[channel_number].append(end7)
            list_of_t5_start[channel_number].append(begin5)
            list_of_t5_end[channel_number].append(end5)
            list_of_t9_start[channel_number].append(begin9)
            list_of_t9_end[channel_number].append(end9)

            list_of_delta_t[channel_number].append((end-begin))
            neutron_amplitude[channel_number].append(maximum_amplitude)
            if ((end - begin)) >= 2:
                number_of_neutron += 1
                # neutron_times[channel_number].append(neutron_time)
                # neutron_event[channel_number].append(binary_packages_checked.index(package))

            # if number_of_neutron_second_method > 0:
            #     waveform_=list(map(lambda y: y-baseline, waveform_tail))
            #     plt.plot(waveform_)
            #     plt.xlim(neutron_time-11, neutron_time+10, 1)
            #     plt.ylim(-2, maximum_amplitude+2)
            #     plt.title("Осциллограмма 'tail' детектора {detector}".format(detector=channel_number+1))
            #     plt.xlabel("Ширина, мкс")
            #     plt.ylabel("А, коды АЦП")
            #     plt.minorticks_on()
            #     plt.grid(which='minor')
            #     plt.grid(which='major')
            #     plt.show()
                # help = re.split('\:|,|\.', time_of_pack)[0]+"_"+re.split('\:|,|\.', time_of_pack)[1]+"_"+\
                #        re.split('\:|,|\.', time_of_pack)[2]
                #
                # plt.savefig("oscillogram{detector}_{time}.png".format(detector=channel_number+1, time=help), dpi=500)
                # plt.close()
            time = int(end) + 1
        else:
            time += 1

    return number_of_neutron

what_it_is = "URAN Test 50k"
os.chdir('//uran/URAN DATA/DATA/Набор данных Серия 3 МГВС/NAD_3/2023/Набор/Тест')
# \\Uran\uran data\DATA\Набор данных Серия 3 МГВС\NAD_3\2022\Набор\Обычный
folder_contents = [path for path in Path.cwd().rglob("*.bin")]
paths = [path for path in folder_contents if pd.to_datetime(re.split(' |,|_', os.path.basename(path))[2]) >= pd.to_datetime('01.01.2023')]
# paths = [path for path in Path.cwd().rglob("*.bin")]
os.chdir("C://Users//funny//PycharmProjects//Working")
sum_packages = 0

for path in paths:
    if sum_packages <= 5000:
        list_of_a = [list() for _ in range(12)]
        list_of_n = [list() for _ in range(12)]
        list_of_baselines = [list() for _ in range(12)]
        list_of_t_start = [list() for _ in range(12)]
        list_of_t_end = [list() for _ in range(12)]
        list_of_t5_start = [list() for _ in range(12)]
        list_of_t5_end = [list() for _ in range(12)]
        list_of_t7_start = [list() for _ in range(12)]
        list_of_t7_end = [list() for _ in range(12)]
        list_of_t9_start = [list() for _ in range(12)]
        list_of_t9_end = [list() for _ in range(12)]
        list_of_amplitude = [list() for _ in range(12)]
        list_of_amplitude5 = [list() for _ in range(12)]
        list_of_amplitude7 = [list() for _ in range(12)]
        list_of_amplitude9 = [list() for _ in range(12)]
        list_of_event_times = [list() for _ in range(12)]
        list_of_delta_t = [list() for _ in range(12)]
        neutron_amplitude = [list() for _ in range(12)]


        with open("{}".format(path), "rb") as file:
            print("Reading file...")
            binary_data = file.read()
        print("Splitting data by packages...")
        binary_packages = split_data_bin_to_packages(binary_data)
        print("Checking distored packages...")
        binary_packages_checked = check_and_remove_distorted_packages(binary_packages)[0]
        print('Number of events =', len(binary_packages_checked))

        for package in tqdm(binary_packages_checked[0:-1]):
                if package == b'':
                    break
                parsed_package = parse_package(package)
                time_of_pack = parsed_package["ts"]
                kr = 0
                for channel_number in range(12):
                    waveform = parsed_package["wf_s"]["ch_{:02}".format(channel_number)]
                    baseline = round(take_baseline(waveform=waveform))
                    maximum = max(waveform) - baseline
                    if maximum >= 10:
                        kr +=1
                if kr >= 0:
                    for channel_number in range(12):
                        waveform = parsed_package["wf_s"]["ch_{:02}".format(channel_number)]
                        waveform_tail = parsed_package["wf_t"]["ch_{:02}".format(channel_number)]
                        baseline = round(take_baseline(waveform=waveform))
                        waveform_ = list(map(lambda y: y - baseline, waveform))
                        number_of_neutrons = (take_number_of_neutron(waveform_tail=waveform_tail, baseline=baseline,
                                                                     channel_number=channel_number, package=package,
                                                                     time_of_pack=time_of_pack))
                        list_of_baselines[channel_number].append(baseline)
                        list_of_a[channel_number].append(max(waveform) - baseline)
                        list_of_n[channel_number].append(number_of_neutrons)

        packages = len(binary_packages_checked)
        sum_packages += len(binary_packages_checked)

        for channel_number in range(12):
            ALL_list_of_a[channel_number] += list_of_a[channel_number]
            ALL_list_of_t_start[channel_number] += list_of_t_start[channel_number]
            ALL_list_of_t_end[channel_number] += list_of_t_end[channel_number]
            ALL_list_of_t5_start[channel_number] += list_of_t5_start[channel_number]
            ALL_list_of_t5_end[channel_number] += list_of_t5_end[channel_number]
            ALL_list_of_t7_start[channel_number] += list_of_t7_start[channel_number]
            ALL_list_of_t7_end[channel_number] += list_of_t7_end[channel_number]
            ALL_list_of_t9_start[channel_number] += list_of_t9_start[channel_number]
            ALL_list_of_t9_end[channel_number] += list_of_t9_end[channel_number]
            ALL_list_of_amplitude[channel_number] += list_of_amplitude[channel_number]
            ALL_list_of_amplitude5[channel_number] += list_of_amplitude5[channel_number]
            ALL_list_of_amplitude7[channel_number] += list_of_amplitude7[channel_number]
            ALL_list_of_amplitude9[channel_number] += list_of_amplitude9[channel_number]
            ALL_list_of_event_times[channel_number] += list_of_event_times[channel_number]


        for channel_number in range(12):
           print(channel_number + 1, 'Среднее число нейтронов', sum(list_of_n[channel_number])/packages)

        folder = re.split('_', os.path.basename(path))[2]+"_{}events".format(len(binary_packages_checked))
        file = save_file(folder,list_of_event_times, list_of_amplitude, list_of_amplitude5, list_of_amplitude7,
                         list_of_amplitude9, list_of_a, list_of_t_start, list_of_t5_start, list_of_t7_start,
                         list_of_t9_start, list_of_t_end, list_of_t5_end, list_of_t7_end, list_of_t9_end)
        # if packages%1000==0:
        #     folder_all = "{what_is_it}_{sum}events".format(what_is_it=what_it_is, sum=sum_packages)
        #
        #     file_save = save_file(folder_all, ALL_list_of_event_times, ALL_list_of_amplitude, ALL_list_of_amplitude5,
        #                           ALL_list_of_amplitude7, ALL_list_of_amplitude9, ALL_list_of_a, ALL_list_of_t_start,
        #                           ALL_list_of_t5_start, ALL_list_of_t7_start, ALL_list_of_t9_start, ALL_list_of_t_end,
        #                           ALL_list_of_t5_end, ALL_list_of_t7_end, ALL_list_of_t9_end)

        print(sum_packages)
    else:
        break

folder_all = "{what_is_it}_{sum}events".format(what_is_it=what_it_is,sum=sum_packages)

file_save = save_file(folder_all,ALL_list_of_event_times, ALL_list_of_amplitude, ALL_list_of_amplitude5,
                      ALL_list_of_amplitude7, ALL_list_of_amplitude9, ALL_list_of_a, ALL_list_of_t_start,
                      ALL_list_of_t5_start, ALL_list_of_t7_start, ALL_list_of_t9_start, ALL_list_of_t_end,
                      ALL_list_of_t5_end, ALL_list_of_t7_end, ALL_list_of_t9_end)
print(file_save)


