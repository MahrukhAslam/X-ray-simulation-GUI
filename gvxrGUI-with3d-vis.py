#!/usr/bin/env python3


# System libs
import os
import sys
import numpy as np


# Matplotlib
import matplotlib

from QSampleViewer import QSampleViewer
from QSampleViewer import USE_QT_VERSION

# Qt
if USE_QT_VERSION == 5:
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    matplotlib.use('Qt5Agg')

elif USE_QT_VERSION == 6:
    from PyQt6.QtGui import *
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    matplotlib.use('QtAgg')

else:
    raise ValueError("Invalid version of PyQt (" + str(USE_QT_VERSION) + "). Only versions 5 and 6 are supported")


from matplotlib.colors import PowerNorm # Look up table
from matplotlib.colors import LogNorm # Look up table


from matplotlib.figure import Figure


from QSampleViewer import QSampleViewer

import json2gvxr
import gvxrPython3 as gvxr


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=500, height=400, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.energy_set = []
        self.count_set = []
        self.raw_x_ray_image = np.zeros((400, 400))
        self.flat_x_ray_image = np.zeros((400, 400))
        self.negative_x_ray_image = np.zeros((400, 400))
        self.screenshot = np.zeros((400, 400))

        self.setWindowTitle("X-ray Simulation ")


        self.create_menu()

        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)
        self.vtk_widget = None

    def create_menu(self):
        main_menu = self.menuBar()
        fileMenu = main_menu.addMenu("File")
        help = main_menu.addMenu("Help")

        newAction = QAction(QIcon('images/new.png'), "New", self)
        newAction.setShortcut("Ctrl+N")
        fileMenu.addAction(newAction)

        
        openAction = QAction("Open", self)
        openAction.setShortcut("Ctrl+O")
        openAction.triggered.connect(self.getJsonFile)
        fileMenu.addAction(openAction)

        fileMenu.addSeparator()

        saveXRayImageAction = QAction(QIcon('images/save.png'), "Save X-ray image", self)
        saveXRayImageAction.setShortcut("Ctrl+S")
        fileMenu.addAction(saveXRayImageAction)

        saveJsonAction = QAction(QIcon('images/save.png'), "Save JSON", self)
        fileMenu.addAction(saveJsonAction)

        saveVizAction = QAction(QIcon('images/save.png'), "Save 3D visualisation", self)
        saveVizAction.triggered.connect(self.saveViz)
        fileMenu.addAction(saveVizAction)

        fileMenu.addSeparator()

        exitAction = QAction(QIcon('images/exit.png'), "Quit", self)
        exitAction.setShortcut("Ctrl+Q")
        exitAction.triggered.connect(self.close_window)
        fileMenu.addAction(exitAction)

    def closeEvent(self, event):
        gvxr.destroyAllWindows()

    def close_window(self):
        gvxr.destroyAllWindows()
        self.close()

    def saveViz(self):
        # The VTK widget exists
        if self.vtk_widget is not None:

            # Get the current directory
            dir_path = os.path.dirname(os.path.realpath(__file__))

            # Open a file dialog box
            file_dialog_box = QFileDialog()

            # Retrieve the filename
            fname = file_dialog_box.getSaveFileName(self, 'Save 3D visualisation',
                                               dir_path ,"PNG File (*.png)")

            if isinstance(fname, tuple):
                png_file_name = fname[0]
            else:
                png_file_name = str(fname)

            # A file name was provided
            if len(png_file_name) > 0:
                self.vtk_widget.takeScreenshot(png_file_name)

    def displayXRayImage(self):

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        sc = MplCanvas(self, width=500, height=500, dpi=100)
        # sc.axes.imshow(self.raw_x_ray_image, cmap="gray")
        sc.axes.imshow(self.flat_x_ray_image,
            cmap="gray",
            norm=LogNorm(vmin=0.01, vmax=1.2))
        # sc.axes.imshow(self.negative_x_ray_image)
        self.table_widget.tab4.layout.addWidget(sc)


    def displayBeamSpectrum(self):

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        sc = MplCanvas(self, width=500, height=500, dpi=100)

        energy_set = np.array(gvxr.getEnergyBins("keV"))
        min_val = energy_set.min()
        max_val = energy_set.max()
        count_set = np.array(gvxr.getPhotonCountEnergyBins())
        sorted_index = np.argsort(energy_set)

        sc.axes.bar(energy_set[sorted_index], count_set[sorted_index] / count_set.sum())
        sc.axes.set_title("Photon energy distribution")
        sc.axes.set_xlabel('Photon energy (in keV)')
        sc.axes.set_ylabel('Probability distribution of photons per keV')

        if min_val != max_val:
            sc.axes.set_xlim([min_val, max_val])
        self.table_widget.tab2.layout.addWidget(sc)

    def displayDetectorEnergyResponse(self):

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        sc = MplCanvas(self, width=500, height=500, dpi=100)
        output_energy = "keV"

        if "Energy response" in json2gvxr.params["Detector"].keys():
            fname = json2gvxr.params["Detector"]["Energy response"]["File"]
            input_energy = json2gvxr.params["Detector"]["Energy response"]["Energy"]
            detector_response = np.loadtxt(fname)
            sc.axes.plot(detector_response[:,0] * gvxr.getUnitOfEnergy(input_energy) / gvxr.getUnitOfEnergy(output_energy),
                detector_response[:,1] * gvxr.getUnitOfEnergy(input_energy) / gvxr.getUnitOfEnergy(output_energy))
        else:
            energy_set = np.array(gvxr.getEnergyBins(output_energy))

            sc.axes.plot([energy_set.min(), energy_set.max()], [energy_set.min(), energy_set.max()])

        sc.axes.set_title("Energy response of the detector")
        sc.axes.set_xlabel('Incident energy: E (in ' + output_energy + ")")
        sc.axes.set_ylabel('Detector energy response: $\\delta$(E) (in ' + output_energy + ")")
        self.table_widget.tab3.layout.addWidget(sc)

    def displayViz(self):

        self.vtk_widget = QSampleViewer()
        self.table_widget.tab1.layout.addWidget(self.vtk_widget)

    def getJsonFile(self):

        # Get the current directory
        dir_path = os.path.dirname(os.path.realpath(__file__))

        # Open a file dialog box
        file_dialog_box = QFileDialog()
        #file_dialog_box.setFileMode(QFileDialog.ExistingFile)

        # Retrieve the filename
        fname = file_dialog_box.getOpenFileName(self, 'open file',
                                           dir_path ,"Json File (*.json)")

        if isinstance(fname, tuple):
            json_file_name = fname[0]
        else:
            json_file_name = str(fname)

        # A file name was provided
        if len(json_file_name) > 0:
            json2gvxr.initGVXR(json_file_name, "EGL")
            gvxr.enableArtefactFilteringOnGPU()
            json2gvxr.initSourceGeometry()
            json2gvxr.initSpectrum()

            energy_set = gvxr.getEnergyBins("MeV")
            count_set = gvxr.getPhotonCountEnergyBins()
            total_energy_in_MeV = 0.0

            for energy, count in zip(energy_set, count_set):
                total_energy_in_MeV += energy * count

            json2gvxr.initSamples(verbose=1)


            json2gvxr.initDetector()


            # Compute the X-ray image
            self.raw_x_ray_image = np.array(gvxr.computeXRayImage())
            print(self.raw_x_ray_image.min(), self.raw_x_ray_image.max())

            # Flat-field
            self.flat_x_ray_image = self.raw_x_ray_image / total_energy_in_MeV
            print(self.flat_x_ray_image.min(), self.flat_x_ray_image.max())

            # Negative
            self.negative_x_ray_image = 1.0 - self.flat_x_ray_image
            print(self.negative_x_ray_image.min(), self.negative_x_ray_image.max())

            # 3D visualisation of the scene
            gvxr.displayScene()

            # Take a screenshot of the visualisation
            self.screenshot = gvxr.takeScreenshot()

            self.displayDetectorEnergyResponse()
            self.displayXRayImage()
            self.displayBeamSpectrum()
            self.displayViz()

            self.show()

class MyTableWidget(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()
        self.tabs.resize(300,200)

        # Add tabs
        self.tabs.addTab(self.tab1,"Sample")
        self.tabs.addTab(self.tab2,"Source")
        self.tabs.addTab(self.tab3,"Detector")
        self.tabs.addTab(self.tab4,"X-ray Image")

        # Add the layout
        self.tab1.layout = QVBoxLayout()
        self.tab1.setLayout(self.tab1.layout)

        self.tab2.layout = QVBoxLayout()
        self.tab2.setLayout(self.tab2.layout)

        self.tab3.layout = QVBoxLayout()
        self.tab3.setLayout(self.tab3.layout)

        self.tab4.layout = QVBoxLayout()
        self.tab4.setLayout(self.tab4.layout)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)



app = QApplication(sys.argv)
w = MainWindow()
w.setMinimumSize(600, 600);
w.show()


app.exec()
gvxr.destroyAllWindows()
