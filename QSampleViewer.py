USE_QT_VERSION = 5

import math

import numpy as np

import matplotlib

from skimage.filters import gaussian # Implementing the image sharpening filter

import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtk.util.numpy_support import numpy_to_vtk

# Qt
if USE_QT_VERSION == 5:
    from PyQt5 import QtWidgets

elif USE_QT_VERSION == 6:
    from PyQt6 import QtWidgets

else:
    raise ValueError("Invalid version of PyQt (" + str(USE_QT_VERSION) + "). Only versions 5 and 6 are supported")

import gvxrPython3 as gvxr # Simulate X-ray images
import json2gvxr # Set gVirtualXRay and the simulation up


def standardisation(img):
    return (img - img.mean()) / img.std()

def logImage(xray_image: np.array, min_val: float, max_val: float) -> np.array:

    log_epsilon = 1.0e-9

    shift_filter = -math.log(min_val + log_epsilon)

    if min_val != max_val:
        scale_filter = 1.0 / (math.log(max_val + log_epsilon) - math.log(min_val + log_epsilon))
    else:
        scale_filter = 1.0

    corrected_image = np.log(xray_image + log_epsilon)

    return (corrected_image + shift_filter) * scale_filter

def applyLogScaleAndNegative(image: np.array) -> np.array:
    temp = logImage(image, image.min(), image.max())
    return 1.0 - temp

def sharpen(image, ksize, alpha, shift, scale):
    details = image - gaussian(image, ksize)

    return scale * (shift + image) + alpha * details

def stl2actor(fname, colour, opacity):

    # Load the file
    reader = vtk.vtkSTLReader()
    reader.SetFileName(fname)
    reader.Update()

    # Get the PolyData
    polydata = vtk.vtkPolyData()
    polydata.ShallowCopy(reader.GetOutput())

    #  Remove any duplicate points.
    clean_filter = vtk.vtkCleanPolyData()

    if vtk.vtkVersion.GetVTKMajorVersion() >= 6:
        clean_filter.SetInputData( polydata )
    else:
        clean_filter.SetInput( polydata )

    clean_filter.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(clean_filter.GetOutputPort())
    # mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colour)
    actor.GetProperty().SetOpacity(opacity)

    print(fname, colour, opacity)
    return actor

class QSampleViewer(QtWidgets.QFrame):
    def __init__(self):

        #Parent constructor
        super(QSampleViewer,self).__init__()

        colors = vtkNamedColors()

        # Make tha actual QtWidget a child so that it can be re parented
        iren = QVTKRenderWindowInteractor(self)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(iren)
        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)

        # Retrieve the detector and source properties
        up = json2gvxr.params["Detector"]["UpVector"]
        src_x_pos, src_y_pos, src_z_pos, src_unit_pos = json2gvxr.params["Source"]["Position"]
        det_x_pos, det_y_pos, det_z_pos, det_unit_pos = json2gvxr.params["Detector"]["Position"]

        # Set a camera that is located on the X-ray source position
        camera = vtk.vtkCamera()
        camera.SetViewUp(up)
        camera.SetPosition(src_x_pos, src_y_pos, src_z_pos)

        # And that looks at the detector
        camera.SetFocalPoint(det_x_pos, det_y_pos, det_z_pos)

        # Set the projection matrix in VTK so that it matches the X-ray projection parameters
        sdd = math.sqrt(math.pow(src_x_pos - det_x_pos, 2) + math.pow(src_y_pos - det_y_pos, 2) + math.pow(src_z_pos - det_z_pos, 2))
        w, h, unit_size = json2gvxr.params["Detector"]["Size"]
        half_w = w / 2
        half_h = h / 2
        half_view_angle_rad = math.atan(half_w / sdd)
        half_view_angle_deg = half_view_angle_rad * 180 / math.pi
        camera.SetViewAngle(2.0 * half_view_angle_deg)

        # Create a rendering window and renderer
        ren = vtk.vtkRenderer()
        ren.SetBackground([1.0, 1.0, 1.0])

        # Use the new camera in VTK
        #ren.SetActiveCamera(camera)

        renWin = iren.GetRenderWindow()
        renWin.AddRenderer(ren)

        iren.SetRenderWindow(renWin)
        renWin.SetInteractor(iren)

        assembly = vtk.vtkAssembly()
        for sample in json2gvxr.params["Samples"]:

            label = sample["Label"]

            fname = sample["Path"]

            r, g, b, a = gvxr.getAmbientColour(label)

            if label == "Muscle":
                opacity = 0.4
            else:
                opacity = 1

            actor = stl2actor(fname, (r, g, b), opacity)
            assembly.AddPart(actor)

        # Add the detector
        detector_actor = self.createDetectorActor()
        # assembly.AddPart(detector_actor)

        # Add the source
        source_actor = self.createSourceActor()
        # assembly.AddPart(source_actor)

        # Assign actors to the renderer
        ren.AddActor(assembly)
        ren.AddActor(detector_actor)
        ren.AddActor(source_actor)

        # Enable user interface interactor
        iren.Initialize()
        renWin.Render()

        self.renderer = ren
        self.render_window = renWin
        self.interactor = iren

    def createSourceActor(self):

        colors = vtkNamedColors()

        src_x_pos, src_y_pos, src_z_pos, src_unit_pos = json2gvxr.params["Source"]["Position"]

        # Create a sphere
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(src_x_pos, src_y_pos, src_z_pos)
        sphereSource.SetRadius(5.0)
        # Make the surface smooth.
        sphereSource.SetPhiResolution(10)
        sphereSource.SetThetaResolution(10)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphereSource.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d("Cornsilk"))

        return actor

    def createDetectorActor(self):

        colors = vtkNamedColors()

        up_x, up_y, up_z = json2gvxr.params["Detector"]["UpVector"]
        # src_x_pos, src_y_pos, src_z_pos, src_unit_pos = json2gvxr.params["Source"]["Position"]
        det_x_pos, det_y_pos, det_z_pos, det_unit_pos = json2gvxr.params["Detector"]["Position"]
        # sdd = math.sqrt(math.pow(src_x_pos - det_x_pos, 2) + math.pow(src_y_pos - det_y_pos, 2) + math.pow(src_z_pos - det_z_pos, 2))

        w, h, unit_size = json2gvxr.params["Detector"]["Size"]
        half_w = w / 2
        half_h = h / 2

        # Define the 4 vertices of the quad
        p0 = [det_x_pos - half_w, det_y_pos, det_z_pos - half_h]
        p1 = [det_x_pos + half_w, det_y_pos, det_z_pos - half_h]
        p2 = [det_x_pos + half_w, det_y_pos, det_z_pos + half_h]
        p3 = [det_x_pos - half_w, det_y_pos, det_z_pos + half_h]

        # Add the points to a vtkPoints object
        points = vtk.vtkPoints()
        points.InsertNextPoint(p0)
        points.InsertNextPoint(p1)
        points.InsertNextPoint(p2)
        points.InsertNextPoint(p3)

        # Create a quad on the four points
        quad = vtk.vtkQuad()
        quad.GetPointIds().SetId(0, 0)
        quad.GetPointIds().SetId(1, 1)
        quad.GetPointIds().SetId(2, 2)
        quad.GetPointIds().SetId(3, 3)

        # Create a cell array to store the quad in
        quads = vtk.vtkCellArray()
        quads.InsertNextCell(quad)


        tcoords = vtk.vtkFloatArray()
        tcoords.SetNumberOfComponents(2)
        tcoords.SetNumberOfTuples(4)
        tcoords.SetTuple2(0, 1.0, 1.0)
        tcoords.SetTuple2(1, 1.0, 0.0)
        tcoords.SetTuple2(2, 0.0, 0.0)
        tcoords.SetTuple2(3, 0.0, 1.0)

        # Create a polydata to store everything in
        polydata = vtk.vtkPolyData()

        # Add the points and quads to the dataset
        polydata.SetPoints(points)
        polydata.SetPolys(quads)
        polydata.GetPointData().SetTCoords(tcoords)

        # Compute the X-ray image
        gvxr.disableArtefactFiltering()
        raw_x_ray_image = np.array(gvxr.computeXRayImage())

        # Flat-field
        total_energy_in_MeV = gvxr.getTotalEnergyWithDetectorResponse()
        white = np.ones(raw_x_ray_image.shape) * total_energy_in_MeV
        dark = np.zeros(raw_x_ray_image.shape)
        flat_x_ray_image = (raw_x_ray_image - dark) / (white - dark)

        # Log-scale
        corrected_xray_image = applyLogScaleAndNegative(flat_x_ray_image)
        corrected_xray_image = corrected_xray_image.astype(np.single)

        # Standardisation
        standardised_corrected_xray_image = standardisation(corrected_xray_image)

        # Sharpening
        sigma1, sigma2, alpha, shift, scale = [3.883818026543349, 1.0000036461307866, 8.223048387996197, 3.097919939169109, 1.9255625476572062]
        sharpened_x_ray_image = standardisation(sharpen(standardised_corrected_xray_image, [sigma1, sigma2], alpha, shift, scale))

        # Min-max normalisation
        temp = sharpened_x_ray_image - sharpened_x_ray_image.min()
        temp /= temp.max()

        # In UINT8
        negative_x_ray_image = (255.0 * temp).astype(np.uint8)

        # Create a vtkImage
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(negative_x_ray_image.shape[1], negative_x_ray_image.shape[0], 1)
        temp1 = np.flip(negative_x_ray_image.swapaxes(0,1), axis=1)
        temp2 = negative_x_ray_image.reshape((-1, 1), order='F')
        print(temp1.shape)
        print(temp2.shape)
        vtkarr = numpy_to_vtk(temp2)
        # vtkarr = numpy_to_vtk(negative_x_ray_image.reshape((-1, 1), order='F'))
        vtkarr.SetName('Image')
        vtk_image.GetPointData().AddArray(vtkarr)
        vtk_image.GetPointData().SetActiveScalars('Image')

        # Create a texture
        vtk_texture = vtk.vtkTexture()
        vtk_texture.SetInputDataObject(vtk_image)
        vtk_texture.InterpolateOn()
        vtk_texture.Update()

        # Setup actor and mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetTexture(vtk_texture)

        # return planeActor
        return actor

    def takeScreenshot(self, fname):
        take_screenshot = vtk.vtkWindowToImageFilter()
        take_screenshot.SetInput(self.render_window)
        take_screenshot.SetInputBufferTypeToRGB()
        take_screenshot.ReadFrontBufferOff()
        take_screenshot.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(fname)
        writer.SetInputConnection(take_screenshot.GetOutputPort())
        writer.Write()

    def start(self):
        self.interactor.Start()
