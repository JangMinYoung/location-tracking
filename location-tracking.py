import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
import math
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def cross_correlation(y1, y2):
    npts = len(y2)

    x = np.linspace(-3, 5, npts)
    y1 = np.array(y1).astype(np.float)
    y2 = np.array(y2).astype(np.float)

    lags = np.arange(-npts + 1, npts)
    ccov = np.correlate(y1 - y1.mean(), y2 - y2.mean(), mode='full')
    ccor = ccov / (npts * y1.std() * y2.std())

    # fig, axs = plt.subplots(nrows=2)
    # fig.subplots_adjust(hspace=0.4)
    # ax = axs[0]
    # ax.plot(x, y1, 'b', label='y1')
    # ax.plot(x, y2, 'r', label='y2')
    # ax.set_ylim(-5, 5)
    # ax.legend(loc='upper right', fontsize='small', ncol=2)
    #
    # ax = axs[1]
    # ax.plot(lags, ccor)
    # ax.set_ylim(-1.1, 1.1)
    # ax.set_ylabel('cross-correlation')
    # ax.set_xlabel('lag of y1 relative to y2')

    maxlag = lags[np.argmax(ccor)]
    print("max correlation is at lag %d" % maxlag)
    result = maxlag / 44100

    # plt.show()
    # plt.close(fig)

    return result


def cal_Altitude(a, td10, td21, td02):
    pow = math.pow(td10, 2) + math.pow(td21, 2) + math.pow(td02, 2)
    pow_sqr = math.sqrt(pow)
    constant = (math.sqrt(2) * 340) / (3 * a)
    # radian=math.radians(constant * pow_sqr)
    mul = constant * pow_sqr
    print("radian:" + str(mul))
    Altitude_angle = math.acos(mul)
    print("Altitude_angle: " + str(Altitude_angle))

    # Altitude_angle2=math.radians(86)
    # print("Altitude_angle2: " + str(Altitude_angle2))

    return Altitude_angle


def cal_Bearing(a, td21, Altitude_angle):
    theta = 0
    constant = (math.sqrt(3) * a) / 340
    b = 8.166603115929353e-05
    # pi_degree = math.degrees(math.pi)
    cos_altitude = math.cos(Altitude_angle)

    x = ((td21 / constant) / cos_altitude)
    # print("x: "+str(x))
    arcoss_value = math.acos(x)

    theta_1 = arcoss_value + (math.pi / 2)
    # theta_2=(math.pi/2)-arcoss_value
    # print(math.degrees(theta_2))

    if math.degrees(theta_1) >= 90:
        theta = 90 - math.degrees(arcoss_value)
    print(theta)
    return math.radians(theta)


def location(r, Altitude_angle, Bearing_angle):
    print("r: ", r)
    x = r * math.cos(Altitude_angle) * math.cos(Bearing_angle)
    y = r * math.cos(Altitude_angle) * math.sin(Bearing_angle)
    z = r * math.sin(Altitude_angle)
    return x, y, z


def graph(x_list, y_list, z_list, a):
    x_list.append(a)
    y_list.append(0)
    z_list.append(0)
    mpl.rcParams['legend.fontsize'] = 10  # 그냥 오른쪽 위에 뜨는 글자크기 설정이다.

    fig = plt.figure()  # 이건 꼭 입력해야한다.
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_list, y_list, z_list, c='b', marker='o', edgecolors='face')  # 위에서 정의한 x,y,z 가지고 그래프그린거다.

    for i in range(len(z_list)):
        print(z_list[i])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()  # 오른쪽 위에 나오는 글자 코드다. 이거 없애면 글자 사라진다. 없애도 좋다.

    plt.show()


def wavefileTransfer(file_count):
    print("=============wavefile->FFT=============")
    for i in range(1, 4):
        fs, data = wavfile.read('./Experiment/' + str(file_count) + '_' + str(i) + '.wav')
        print('샘플링 수 : ', fs)

        text = open('./Experiment/' + str(file_count) + '_' + str(i) + '.txt', 'w')
        for m in range(len(data)):
            text.write(str(data[m]) + "\n")
        print(len(data))
        text.close()


def main():
    x_list = []
    y_list = []
    z_list = []
    a = 2

    for i in range(1, 4):

        file_count = i
        wavefileTransfer(file_count)

        text1 = open('./Experiment/' + str(file_count) + '_1.txt', 'r')
        text2 = open('./Experiment/' + str(file_count) + '_2.txt', 'r')
        text3 = open('./Experiment/' + str(file_count) + '_3.txt', 'r')

        r = 0

        y0 = []
        y1 = []
        y2 = []
        for line in text1:
            y0.append(line)
        for line2 in text2:
            y1.append(line2)
        for line3 in text3:
            y2.append(line3)

        td10 = cross_correlation(y1, y0)
        td21 = cross_correlation(y2, y1)
        td02 = cross_correlation(y0, y2)

        Altitude_angle = cal_Altitude(a, td10, td21, td02)
        Bearing_angle = cal_Bearing(a, td21, Altitude_angle)

        if i == 1:
            r = 127
        elif i == 2:
            r = 254
        elif i == 3:
            r = 105

        x, y, z = location(r, Altitude_angle, Bearing_angle)
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)

        print("x:" + str(x), "y:" + str(y), "z:" + str(z))
    graph(x_list, y_list, z_list, a)


main()
