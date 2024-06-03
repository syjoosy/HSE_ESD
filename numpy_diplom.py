import numpy as np
from stl import mesh
import time
import threading
# import math

import pymesh

print_data = False

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
# from tkinter import *

import sys
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel


# new_point1 = 0
# new_point2 = 0
# new_point3 = 0

# new_point1 = 1.5979159
# new_point2 = 0.005305909
# new_point3 = 0.0

new_point1 = 0.61402583
new_point2 = -0.59795445
new_point3 = -0.08241869


# new_point1 = 0.2636516
# new_point2 = -0.09084277
# new_point3 = -0.30070627

class ExampleApp:

    def __init__(self, cloud):
        # Создадим виджет сцены, который заполнит все окно, а затем
        # метку в левом нижнем углу поверх виджета сцены, чтобы отобразить
        # координаты.
        app = gui.Application.instance
        self.window = app.create_window("Nikolaev Vadim - Diplom", 1024, 768)
        # Поскольку мы хотим, чтобы надпись располагалась поверх сцены, мы не можем использовать макет,
        # поэтому нам нужно вручную расположить дочерние элементы окна.
        self.window.set_on_layout(self._on_layout)
        self.widget3d = gui.SceneWidget()
        self.window.add_child(self.widget3d)
        self.info = gui.Label("")
        self.info.visible = False
        self.window.add_child(self.info)

        
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)

        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        # Размер точки указан в собственных пикселях, но "пиксель" означает разные вещи для
        # разных платформ, поэтому умножим на масштаб окна и коэффициент.
        mat.point_size = 3 * self.window.scaling
        self.widget3d.scene.add_geometry("Point Cloud", cloud, mat)

        bounds = self.widget3d.scene.bounding_box
        center = bounds.get_center()
        self.widget3d.setup_camera(60, bounds, center)
        self.widget3d.look_at(center, center - [0, 0, 3], [0, -1, 0])

        self.widget3d.set_on_mouse(self._on_mouse_widget3d)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.widget3d.frame = r
        pref = self.info.calc_preferred_size(layout_context,
                                             gui.Widget.Constraints())
        self.info.frame = gui.Rect(r.x,
                                   r.get_bottom() - pref.height, pref.width,
                                   pref.height)

    def _on_mouse_widget3d(self, event):
        # Мы могли бы переопределить BUTTON_DOWN без модификатора, но это помешало бы манипулированию сценой.
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                gui.KeyModifier.CTRL):

            def depth_callback(depth_image):
                global new_point1
                global new_point2
                global new_point3
                # Координаты выражены в абсолютных координатах окна
                x = event.x - self.widget3d.frame.x
                y = event.y - self.widget3d.frame.y
                # np.asarray() меняет оси местами.
                depth = np.asarray(depth_image)[y, x]
                
                # Нажали на пустоту
                if depth == 1.0:
                    text = "Мимо"
                    print("Click X-Y: " + str(x) + " " + str(y) + " Depth: " + str(depth))
                else:
                    world = self.widget3d.scene.camera.unproject(
                        x, y, depth, self.widget3d.frame.width,
                        self.widget3d.frame.height)
                    text = "({:.3f}, {:.3f}, {:.3f})".format(
                        world[0], world[1], world[2])

                    print(world[0], world[1], world[2])
                    new_point1 = world[0]
                    new_point2 = world[1]
                    new_point3 = world[2]

                    app.quit()

                    

                # Это не вызывается в основном потоке, поэтому нужно
                # отправить сообщение в основной поток, 
                # чтобы безопасно получить доступ к элементам пользовательского интерфейса.
                def update_label():
                    self.info.text = text
                    self.info.visible = (text != "")
                    # Мы подбираем размер информационной надписи точно под нужный размер,
                    # поэтому нужно изменить макет, чтобы установить новую рамку.
                    self.window.set_needs_layout()

                gui.Application.instance.post_to_main_thread(
                    self.window, update_label)

            self.widget3d.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED



def stl_to_point(v1, v2, v3, num_points, sampling_mode="weight"):
    """
    Функция для преобразования stl-файла в облако точек
    :параметры v1, v2, v3 : (N,3) ndarrays, vi представляют координаты x,y,z одной вершины
    :параметр num_points: Количество точек, которые мы хотим отобрать
    :параметр sampling_mode: Строка, тип выборки из треугольника, рекомендуемый "вес"
    :return: points: массив numpy из облака точек
    """
    if not (np.shape(v1)[0] == np.shape(v2)[0] == np.shape(v3)[0]):
        raise ValueError("Size of all three vertex is not the same")
    else:
        if print_data:
            print("Number of mesh: %s" % np.shape(v1)[0])
    areas = triangle_area_multi(v1, v2, v3)
    prob = areas / areas.sum()
    if sampling_mode == "weight":
        indices = np.random.choice(range(len(areas)), size=num_points, p=prob)
    else:
        indices = np.random.choice(range(len(areas)), size=num_points)
    points = select_point_from_triangle(v1[indices, :], v2[indices, :], v3[indices, :])
    return points

def sample_points_evenly(v1, v2, v3, edge_length, desired_spacing):
    num_points_per_edge = int(round(edge_length / desired_spacing)) + 1
    total_points = (num_points_per_edge - 1) ** 2
    print_data = True
    sampling_mode = "weight"
    points = stl_to_point(v1, v2, v3, total_points, sampling_mode=sampling_mode)
    return points

def triangle_area_multi(v1, v2, v3):
    """
    Найдите площадь треугольника, используемую для определения весов
    :параметры v1, v2, v3 : (N,3) ndarrays, vi представляют координаты x,y,z одной вершины
    :return: размер треугольника
    """
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)


def select_point_from_triangle(v1, v2, v3):
    """
    Выберите одну точку из каждых трех вершин
    :параметры v1, v2, v3 : (N,3) ndarrays, vi представляют координаты x,y,z одной вершины
    ::return: ndarray
    """
    n = np.shape(v1)[0]
    u = np.random.rand(n, 1)
    v = np.random.rand(n, 1)
    is_a_problem = u + v > 1

    u[is_a_problem] = 1 - u[is_a_problem]
    v[is_a_problem] = 1 - v[is_a_problem]

    w = 1 - (u + v)

    points = (v1 * u) + (v2 * v) + (v3 * w)

    return points

def run_app(app):
    app.run_in_thread()

def drawScheme(x, y, z):
    global pcd
    point = stl_to_point(a.v0, a.v1, a.v2, 1000000, sampling_mode="weight")

    #print(type(point))


    # ваш двумерный массив
    # point

    # расстояние между точкой и точкой с координатами (0.5, 0)
    distances = np.linalg.norm(point - np.array([[x, y, z]]), axis=1)

    result_array = np.empty((0,3))

    count = 0.99

    for i in np.arange(0.02, 15, 0.02):
        print(i)
        # условие: расстояние меньше 
        condition = distances < i

        # перебор массива и оставление только тех точек, которые удовлетворяют условию
        filtered_arr = point[np.where(condition)]

        n = filtered_arr.shape[0]

        # Определим количество элементов для выборки (99% от общего количества элементов)
        if (count < 0):
            count = 0.0001
        n_selected = int(n * count)

        selected_indices = np.random.choice(n, size=n_selected, replace=False)
        selected_elements = filtered_arr[selected_indices]
        # random_elements = np.random.choice(filtered_arr, len(filtered_arr)/2, replace=False)

        result_array = np.concatenate((result_array, selected_elements))
        # if (count < 0.1):
        #     count -= 0.0015
        # else:
        
        count -= 0.01


    

    print(filtered_arr)
    
    print("Visualize:")
       
    #pcd.points = o3d.utility.Vector3dVector(result_array)
    #app.post_to_main_thread(ex.get_geometry_name(), pcd)

    app = gui.Application.instance
    app.initialize()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(result_array)
    ex = ExampleApp(pcd)

    # app_thread = threading.Thread(target=run_app, args=(app,))
    # app_thread.start()
    # app.run_in_thread()
    app.run()



def view_3d_point_cloud(point):
    # %matplotlib inline
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])
    ax.scatter3D(point[:, 0], point[:, 2], point[:, 1], cmap='Greens')

file = ""

def dialog():
    global file
    file , check = QtWidgets.QFileDialog.getOpenFileName(None, "QFileDialog.getOpenFileName()",
                                               "", "STL (*.stl)")

    if check:
        QtWidgets.qApp.quit()


def create_grid(points, spacing=20):
    # Разделение точек на X, Y и Z
    x, y, z = np.array(points).T

    # нахождение минимального и максимального значения X, Y и Z
    x_min, y_min, z_min = np.min(x), np.min(y), np.min(z)
    x_max, y_max, z_max = np.max(x), np.max(y), np.max(z)

    # Создание сетки
    x_grid = np.arange(x_min, x_max, spacing)
    y_grid = np.arange(y_min, y_max, spacing)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)

    # Выравнивание сетки
    z_grid = np.zeros(x_grid.shape)
    indices = np.indices(x_grid.shape)
    z_grid[indices] = z[indices]

    # Создание нового массива точек со значениями сетки.
    grid_points = np.column_stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()])

    return grid_points

def mesh_to_point_cloud(mesh, spacing):
    points = mesh.vertices.copy()
    point_cloud = []
    for i in range(points.shape[0]):
        point_cloud.append(points[i])
        if i < points.shape[0] - 1:
            next_point = points[i + 1]
            if np.linalg.norm(points[i] - next_point) < spacing:
                continue
        point_cloud.append(points[i] + np.array([spacing, 0, 0]))
        point_cloud.append(points[i] + np.array([0, spacing, 0]))
        point_cloud.append(points[i] + np.array([0, 0, spacing]))

    return np.array(point_cloud)

def visualize_point_cloud(point_cloud):
    import open3d as o3d
    pcd = o3d.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pcd])

def distance(point1, point2):
    return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(point1, point2)))

def filter_points(points, distance_threshold):
    filtered_points = [points[0]]
    for point in points[1:]:
        for filtered_point in filtered_points:
            if distance(point, filtered_point) >= distance_threshold:
                filtered_points.append(point)
                break
    return filtered_points

def find_closest_point(points, point):
    """
    Нахождение точки в массиве, которая находится ближе всего к заданной точке.
    """
    distances = np.linalg.norm(points - point, axis=1)
    return np.argmin(distances)

def remove_close_points(points, distance):
    """
    Удаление точек, которые находятся ближе друг к другу, чем указанное расстояние.
    """
    while len(points) > 1:
        point = points[0]
        # Нахождение ближайшей точки к текущей точке
        closest_point_index = find_closest_point(points[1:], point)
        closest_point = points[closest_point_index + 1]
        # Вычисление расстояния между точками
        distance = np.linalg.norm(closest_point - point)
        if distance < distance:
            # Удаление точки
            points = np.delete(points, closest_point_index + 1, axis=0)
        else:
            break

    return points

def is_point_within_distance(point, points, d):
    """
    Проверяет, находится ли точка на расстоянии d от любой другой точки в массиве.
    
    :param point: Точка, для которой проверяется расстояние.
    :param points: Массив точек.
    :param d: Расстояние, на котором проверяется нахождение точки.
    :return: True, если точка находится на расстоянии d от любой другой точки, иначе False.
    """
    for other_point in points:
        if np.linalg.norm(point - other_point) <= d:
            return False
    return True

def filter_points_by_distance(points, d):
    """
    Фильтрует точки из массива, оставляя только те, которые находятся на расстоянии d друг от друга.
    
    :param points: Массив точек.
    :param d: Расстояние, на котором проверяется нахождение точки.
    :return: Массив точек, которые находятся на расстоянии d друг от друга.
    """
    filtered_points = []
    for point in points:
        if is_point_within_distance(point, points, d):
            filtered_points.append(point)
    return filtered_points

def remove_close_points2(points, distance):
    # Сортировка точек по координате x
    points = np.sort(points, axis=0)

    # Инициализируем пустой массив для хранения отфильтрованных точек
    filtered_points = np.empty((0, 3))

    for i in range(len(points) - 1):
        # Проверяем расстояние между текущей точкой и следующей
        dx = points[i+1, 0] - points[i, 0]
        dy = points[i+1, 1] - points[i, 1]
        dz = points[i+1, 2] - points[i, 2]
        d = np.sqrt(dx**2 + dy**2 + dz**2)

        # Если расстояние больше или равно указанному расстоянию, добавляем текущую точку в отфильтрованный массив.
        if d >= distance:
            filtered_points = np.vstack((filtered_points, points[i]))

    # Добавляем последнюю точку в отфильтрованный массив
    filtered_points = np.vstack((filtered_points, points[-1]))

    return filtered_points


if __name__ == "__main__":
    

    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(100,200,300,100)
    win.setWindowTitle("Nikolaev Vadim - Diplom")

    # label= QLabel(win)
    # label.setText("Hi this is Pyqt5")
    # label.move(100,0)

    # textbox = QtWidgets.QLineEdit(win)
    # textbox.move(100, 100)
    # textbox.resize(180,40)

    # line = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, win)
    # line.move(100,100)

    button2 = QtWidgets.QPushButton(win)
    button2.setText("Choose 3D model")
    button2.clicked.connect(dialog)
    button2.move(50,25)
    button2.resize(180,40)

    win.show()
    app.exec_()

    if file != "":
        print("File: ")
        print(file)
    else:
        print("No File Selected")
        sys.exit()

    # file = "/Users/alex/Documents/Diplom/python/2x2.stl"
    # # load the stl file
    # mesh = pymesh.load_mesh(file)
    # # convert the mesh to a point cloud
    # point_cloud = mesh_to_point_cloud(mesh, 0.2) # 20 cm
    # # visualize the point cloud
    # visualize_point_cloud(point_cloud)

    a = mesh.Mesh.from_file(file)
    # a = trimesh.load(file)
    # if 
    # sys.exit(app.exec_())

    # window = Tk()
    # window.title("Добро пожаловать в приложение PythonRu")
    # window.mainloop()

    # v1 = np.array([[1, 0, 0], [2, 3, 1], [-2, 5, 0]])
    # v2 = np.array([[0, 0, 0], [2, 4, 2], [0, 5, 4]])
    # v3 = np.array([[0, 1, 0], [2, 4, 1], [2, 5, 0]])
    print_data = True
    edge_length = 1.0
    # point = sample_points_evenly(a.v0, a.v1, a.v2, edge_length, 0.2)
    point = stl_to_point(a.v0, a.v1, a.v2, 10000000, sampling_mode="weight")

    print(point)

    # distance = 1.5

    # filtered_points = remove_close_points2(point, 100.1)
    
    # filtered_points = filter_points_by_distance(point, distance)

    # filtered_points = remove_close_points(point, 500.0)
    # filtered_points = [p for p in point if any(is_point_within_distance(p, q, distance) for q in point if q != p)]

    # distances = np.linalg.norm(point - np.array([[new_point1, new_point2, new_point3]]), axis=1)

    # condition = distances < 0.25


    # # перебор массива и оставление только тех точек, которые удовлетворяют условию
    # filtered_arr = point[np.where(condition)]

    # point = filtered_points

    # Define the distance between points in meters (0.2 meters = 20 cm)
    # desired_distance = 0.1
    # point = []
    # for i in range(len(a.vectors)):
    #     points = a.vectors[i]
    #     point.append(points)
    #     point.append(points + np.array([20, 0, 0])) # Add points with 20 cm X-axis shift
    #     point.append(points + np.array([0, 20, 0])) # Add points with 20 cm Y-axis shift
    #     point.append(points + np.array([0, 0, 20])) # Add points with 20 cm Z-axis shift

    # point = np.array(point)
    # Calculate the number of points needed for the given distance\
    # num_points = int(np.floor(np.linalg.norm(a.v0[0, :]) / desired_distance)) + 1

    # distance = 0.2  # 20 cm
    # vol = a.volume
    # r = (3 * vol / (4 * np.pi))**(1/3)  # radius of a sphere with the same volume
    # n_points = int(4/3 * np.pi * r**3 / (4/3 * np.pi * (distance/2)**3))

    # Generate the point cloud
    # point_cloud = a.uniform_points(n_points)
    
    # point = point_cloud
    
    # point = stl_to_point(a.v0, a.v1, a.v2, num_points, sampling_mode="weight")

    # point = stl_to_point(a.v0, a.v1, a.v2, 10000, sampling_mode="weight")

    # point_grid = create_grid(point)

    # point = point_grid

    # distances = np.sqrt(np.sum((point[:, np.newaxis, :] - point[np.newaxis, :, :]) ** 2, axis=2))
    
    # point = point[np.where(distances == 20)]
    

    
    
    
    
    
    
    
    # point = stl_to_point(a.v0, a.v1, a.v2, 1)

    #print(type(point))


    # ваш двумерный массив
    # point

    # расстояние между точкой и точкой с координатами (0.5, 0)
    # distances = np.linalg.norm(point - np.array([[1.2656598, -3.341467, 0.15297416]]), axis=1)

    #result_array = np.empty((0,3))

    #count = 0.99

    # for i in np.arange(0.02, 15, 0.02):
    #     print(i)
    #     # условие: расстояние меньше 
    #     condition = distances < i

    #     # перебор массива и оставление только тех точек, которые удовлетворяют условию
    #     filtered_arr = point[np.where(condition)]

    #     n = filtered_arr.shape[0]

    #     # Определим количество элементов для выборки (99% от общего количества элементов)
    #     if (count < 0):
    #         count = 0.0001
    #     n_selected = int(n * count)

    #     selected_indices = np.random.choice(n, size=n_selected, replace=False)
    #     selected_elements = filtered_arr[selected_indices]
    #     # random_elements = np.random.choice(filtered_arr, len(filtered_arr)/2, replace=False)

    #     result_array = np.concatenate((result_array, selected_elements))
    #     # if (count < 0.1):
    #     #     count -= 0.0015
    #     # else:
        
    #     count -= 0.01


    

    # print(filtered_arr)
    
    print("Visualize:")

    app = gui.Application.instance
    app.initialize()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point)
    

    colors = np.full((point.shape[0], 3), [1, 0.5, 0])  # красный цвет

    # num_points = point.shape[0]
    # colors = np.zeros((num_points, 3))
    # colors[:, 0] = 1  # R
    # colors[:, 1] = 0  # G
    # colors[:, 2] = 0  # B
    pcd.colors = o3d.utility.Vector3dVector(colors)

    #colors = np.full((point.shape[0], 3), [0, 1, 0])  # красный цвет

    #pcd.colors = o3d.utility.Vector3dVector(colors)

    #pcd.colors = o3d.utility.Vector3dVector(colors)
    
    #pcd.points = o3d.utility.Vector3dVector(point[:1000])

    #pcd2 = o3d.geometry.PointCloud()
    #pcd2.points = o3d.utility.Vector3dVector(point[1000:1000000])

    
    # Paint points with different colors
    #pcd.paint_uniform_color([1, 0, 0])  # first point is red
    #pcd2.paint_uniform_color([0, 1, 0])  # second point is green
    # pcd.paint_uniform_color([0, 0, 1])



   
    ex = ExampleApp(pcd)

    #app.run()

    # app_thread = threading.Thread(target=run_app, args=(app,))
    # app_thread.start()
    app.run()

    if (new_point1 != 0.0 or new_point2 != 0.0 or new_point3 != 0.0):

        print("Start new app")

        print(new_point1)
        print(new_point2)
        print(new_point3)

        # drawScheme(new_point1, new_point2, new_point3)]
                                              #1000000
        point = stl_to_point(a.v0, a.v1, a.v2, 100000, sampling_mode="weight")



        # расстояние между точкой и точкой с координатами (0.5, 0)
        distances = np.linalg.norm(point - np.array([[new_point1, new_point2, new_point3]]), axis=1)

        result_array = np.empty((0,3))

        count = 0.99
        
        test = True

        app = gui.Application.instance
        app.initialize()
        pcd = o3d.geometry.PointCloud()

        for i in np.arange(0.02, 15, 0.02):
            print(str(int((i - 0.02) / (15-0.02) * 100)) + "%")
            # условие: расстояние меньше 
            condition = distances < i

            # перебор массива и оставление только тех точек, которые удовлетворяют условию
            filtered_arr = point[np.where(condition)]

            n = filtered_arr.shape[0]

            # Определим количество элементов для выборки (99% от общего количества элементов)
            if (count < 0.5):
                count = 0.0001

            n_selected = int(n * count)

            selected_indices = np.random.choice(n, size=n_selected, replace=False)
            selected_elements = filtered_arr[selected_indices]
   

            result_array = np.concatenate((result_array, selected_elements))

            count -= 0.005

        
        colors = np.zeros((result_array.shape[0], 3))
        # colors = np.full((point.shape[0], 3), [1, 0.5, 0])  # красный цвет

        for i in range(result_array.shape[0]):
            distance = np.linalg.norm(result_array[i] - np.array([[new_point1, new_point2, new_point3]]))

            if (distance <= 0.55):
                colors[i] = [1, 0, 0]
                continue
            elif (distance <= 0.9):
                colors[i] = [1, 0.4, 0]
                continue
            elif (distance <= 1.25):
                colors[i] = [1, 0.8, 0]
                continue
            elif (distance <= 2.3):
                colors[i] = [0, 0.5, 0]
                continue
            else:
                colors[i] = [0, 0, 0.5]

        
        
        # 
        # colors = np.full((0, 3), [0, 1, 0])  # красный цвет

        # for i in np.arange(0.02, 15, 0.02):
        #     print(str(int((i - 0.02) / (15-0.02) * 100)) + "%")
        #     # условие: расстояние меньше 
        #     condition = distances < i

        #     filtered_arr = result_array[np.where(condition)]

            #red_color = np.full((filtered_arr.shape[0], 3), [1.0, 0.0, 0.0])

            #colors = np.concatenate(colors, np.array(red_color, dtype=np.float32))
            #colors = np.concatenate([colors.reshape(-1, 3), red_color], axis=0)

            #if (i >= 5):
            #    break
            # colors 





        # test2 = 0

        # n = len(result_array) // 2 

        # for i in range(len(result_array)):
        #     print (str(i) + "/" + str(len(result_array)))
        #     colors[i] = [1,0,0]

        #     test2 += 1

        #     if (test2 >= 10000000):
        #         break



        # for i in np.arange(0.02, 15, 0.02):
        #     # print (str(i) + "/" + str(len(result_array)))
        #     #print(str(int((i - 0.02) / (15-0.02) * 100)) + "%")
        #     condition = distances < i
        #     # # перебор массива и оставление только тех точек, которые удовлетворяют условию
        #     filtered_arr = result_array[np.where(condition)]
        #     # # if (i < 5):
        #     print("Count of red points: " + str(len(filtered_arr)))
        #     for j in range(len(filtered_arr)):
        #         if (test):
        #             colors[j] = [1,0,0]
        #             test = False
        #         else:
        #             colors[j] = [0,0,1]
            
        #     test2 += 1

        #     if (test2 >= 100):
        #         break
            # else:
            # break

        # red_points = np.arange(0, 100)
        # colors[red_points, 0] = 1.0  # R
        # colors[red_points, 1] = 0.0  # G
        # colors[red_points, 2] = 0.0  # B
        
        # pcd.colors = o3d.utility.Vector3dVector(colors)

        print(filtered_arr)
        
        print("Visualize:")
        
        #pcd.points = o3d.utility.Vector3dVector(result_array)
        #app.post_to_main_thread(ex.get_geometry_name(), pcd)

        
        



        pcd.points = o3d.utility.Vector3dVector(result_array)


        # distances = np.linalg.norm(pcd.points, axis=1)
        # colors = o3d.utility.Vector3dVector(np.stack([distances / distances.max(), 
        #                                             (distances / distances.max())**0.5, 
        #                                             (distances / distances.max())**2], axis=1))

        pcd.colors = o3d.utility.Vector3dVector(colors)

        # pcd.colors = colors

        ex = ExampleApp(pcd)

        # app_thread = threading.Thread(target=run_app, args=(app,))
        # app_thread.start()
        # app.run_in_thread()
        app.run()
    else:
        print("No start app")
        print(new_point1)
        print(new_point2)
        print(new_point3)


    # # Convert to open3d PointCloud
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(result_array)

    # # Visualize the point cloud
    # # o3d.visualization.draw_geometries([pcd])

    # vis = o3d.visualization.VisualizerWithEditing()
    # vis.create_window()
    # vis.add_geometry(pcd)
    # vis.run()  # user picks points
    

    # print("Picked point: ")

    # print(vis.get_picked_points())

    # vis.destroy_window()
    # print(point)