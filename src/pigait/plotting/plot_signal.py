import ahrs
import matplotlib.pyplot as plt
import numpy as np

from ..data import event_data


# Plot events
def plot_wavelet_events(data, acc_ap_pp, acc_wave_detrended, acc_wave_2):
    time_vector = data.time_vector
    hs_idx = [event.sample_idx for event in data.events
              if event.type == event_data.GaitEventType.HEEL_STRIKE]
    to_idx = [event.sample_idx for event in data.events
              if event.type == event_data.GaitEventType.TOE_OFF]
    inval_hs_idx = [event.sample_idx for event in data.events
                    if (event.type == event_data.GaitEventType.HEEL_STRIKE
                        and event.validity != event_data.GaitEventValidity.VALID)]
    hs_times = time_vector[hs_idx]
    to_times = time_vector[to_idx]
    inval_hs_times = time_vector[inval_hs_idx]
    plt.figure(0)
    plt.plot(time_vector, acc_ap_pp, 'b--', label='acc_ap')
    plt.plot(time_vector, acc_wave_detrended, color="grey",
             linestyle="dotted", label='dcwt1')
    plt.plot(time_vector, acc_wave_2, 'k', label='dcwt2')
    plt.plot(hs_times, acc_wave_detrended[hs_idx],
             linestyle='None', color='red', label='HS',
             marker="*",  markersize=10)
    plt.plot(to_times, acc_wave_2[to_idx],
             linestyle='None', color='lime', label='TO',
             marker="*",  markersize=10)
    plt.plot(inval_hs_times, acc_wave_detrended[inval_hs_idx],
             linestyle='None', color='orange', label='HS invalid',
             marker="*",  markersize=10)
    plt.xlabel('Time [s]')
    plt.legend(loc="upper right")
    plt.show()


def plot_position(data):
    # Get events to calculate positions between
    hs_lf = data.get_events(event_data.GaitEventType.HEEL_STRIKE,
                            event_data.GaitEventSide.LEFT)
    hs_lf = [event.sample_idx for event in hs_lf]
    hs_rf = data.get_events(event_data.GaitEventType.HEEL_STRIKE,
                            event_data.GaitEventSide.RIGHT)
    hs_rf = [event.sample_idx for event in hs_rf]
    all_hs = np.sort(np.concatenate((hs_lf, hs_rf)))
    plt.figure(0)
    ax = plt.axes(projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.plot3D(data.position[:, 0], data.position[:, 1], data.position[:, 2],
              'gray')
    ax.plot3D(data.position[:, 0][all_hs], data.position[:, 1][all_hs],
              data.position[:, 2][all_hs], 'k*')
    ax.set_xlabel('X (East)')
    ax.set_ylabel('Y (North)')
    ax.set_zlabel('Z (Up)')
    set_axes_equal(ax)
    plt.show()


def plot_velocity(data):
    vel = data.velocity
    plt.figure(0)
    t = np.arange(vel.shape[0])
    plt.plot(t, vel[:, 0], label='x')
    plt.plot(t, vel[:, 1], label='y')
    plt.plot(t, vel[:, 2], label='z')
    plt.title('velocity')
    plt.legend()
    plt.show()


# Plot two data with HS
def plot_axes_with_hs(data, data_1, data_2, label):
    hs_lf = data.get_events(event_data.GaitEventType.HEEL_STRIKE,
                            event_data.GaitEventSide.LEFT)
    hs_lf = [event.sample_idx for event in hs_lf]
    hs_rf = data.get_events(event_data.GaitEventType.HEEL_STRIKE,
                            event_data.GaitEventSide.RIGHT)
    hs_rf = [event.sample_idx for event in hs_rf]
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    t = np.arange(data_1.shape[0])
    ax1.plot(t, data_1[:, 0], label='X 1')
    ax1.plot(t, data_2[:, 0], label='X 2')
    ax1.plot(hs_lf, data_2[:, 0][hs_lf], 'r*', label='HS lf')
    ax1.plot(hs_rf, data_2[:, 0][hs_rf], 'c*', label='HS rf')
    ax1.legend(loc="upper right")
    ax1.set_title(label + ' X')
    ax2.plot(t, data_1[:, 1], label='Y 1')
    ax2.plot(t, data_2[:, 1], label='Y 2')
    ax2.plot(hs_lf, data_2[:, 1][hs_lf], 'r*', label='HS lf')
    ax2.plot(hs_rf, data_2[:, 1][hs_rf], 'c*', label='HS rf')
    ax2.legend(loc="upper right")
    ax2.set_title(label + ' Y')
    ax3.plot(t, data_1[:, 2], label='Z 1')
    ax3.plot(t, data_2[:, 2], label='Z 2')
    ax3.plot(hs_lf, data_2[:, 2][hs_lf], 'r*', label='HS lf')
    ax3.plot(hs_rf, data_2[:, 2][hs_rf], 'c*', label='HS rf')
    ax3.legend(loc="upper right")
    ax3.set_title(label + ' Z')


# Plot a quaternion converted to euler angles
def plot_euler(ax, q, ff, title_str):
    euler_angles = np.array(
        [ahrs.Quaternion(q_arr).to_angles() for q_arr in q])
    euler_angles = np.degrees(euler_angles)
    t = np.arange(euler_angles.shape[0])
    ax.plot(t, euler_angles[:, 0], label='x')
    ax.plot(t, euler_angles[:, 1], label='y')
    ax.plot(t, euler_angles[:, 2], label='z')
    ax.plot(ff, euler_angles[:, 0][ff], 'k*')
    ax.plot(ff, euler_angles[:, 1][ff], 'k*')
    ax.plot(ff, euler_angles[:, 2][ff], 'k*')
    ax.set_xlabel('Sample', fontsize=15)
    ax.set_title(title_str, size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=13)
    ax.legend(loc="upper right")


# Plot a quaternion
def plot_quaternions(ax, q, ff, title_str):
    t = np.arange(q.shape[0])
    ax.plot(t, q[:, 0], label='w')
    ax.plot(t, q[:, 1], label='x')
    ax.plot(t, q[:, 2], label='y')
    ax.plot(t, q[:, 3], label='z')
    ax.plot(ff, q[:, 0][ff], 'k*')
    ax.plot(ff, q[:, 1][ff], 'k*')
    ax.plot(ff, q[:, 2][ff], 'k*')
    ax.plot(ff, q[:, 3][ff], 'k*')
    ax.set_xlabel('Sample', fontsize=15)
    ax.set_ylim(-1, 1)
    ax.set_title(title_str, size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=13)
    ax.legend(loc="upper right")


def plot_filter(gyro_data, acc_data, mag_data,
                gyro_data_filt, acc_data_filt, mag_data_filt):

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    t = np.arange(gyro_data.shape[0])
    ax1.plot(t, gyro_data[:, 0], label='X orig')
    ax1.plot(t, gyro_data_filt[:, 0], label='X filt')
    ax1.legend(loc="upper right")
    ax1.set_title('Gyro X filt')
    ax2.plot(t, gyro_data[:, 1], label='Y orig')
    ax2.plot(t, gyro_data_filt[:, 1], label='Y filt')
    ax2.legend(loc="upper right")
    ax2.set_title('Gyro Y filt')
    ax3.plot(t, gyro_data[:, 2], label='Z orig')
    ax3.plot(t, gyro_data_filt[:, 2], label='Z filt')
    ax3.legend(loc="upper right")
    ax3.set_title('Gyro Z filt')
    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    t = np.arange(acc_data.shape[0])
    ax1.plot(t, acc_data[:, 0], label='X orig')
    ax1.plot(t, acc_data_filt[:, 0], label='X filt')
    ax1.legend(loc="upper right")
    ax1.set_title('Acc X filt')
    ax2.plot(t, acc_data[:, 1], label='Y orig')
    ax2.plot(t, acc_data_filt[:, 1], label='Y filt')
    ax2.legend(loc="upper right")
    ax2.set_title('Acc Y filt')
    ax3.plot(t, acc_data[:, 2], label='Z orig')
    ax3.plot(t, acc_data_filt[:, 2], label='Z filt')
    ax3.legend(loc="upper right")
    ax3.set_title('Acc Z filt')
    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    t = np.arange(mag_data.shape[0])
    ax1.plot(t, mag_data[:, 0], label='X orig')
    ax1.plot(t, mag_data_filt[:, 0], label='X filt')
    ax1.legend(loc="upper right")
    ax1.set_title('Mag X filt')
    ax2.plot(t, mag_data[:, 1], label='Y orig')
    ax2.plot(t, mag_data_filt[:, 1], label='Y filt')
    ax2.legend(loc="upper right")
    ax2.set_title('Mag Y filt')
    ax3.plot(t, mag_data[:, 2], label='Z orig')
    ax3.plot(t, mag_data_filt[:, 2], label='Z filt')
    ax3.legend(loc="upper right")
    ax3.set_title('Mag Z filt')
    plt.show()


# Helper function for plotting 3D
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
