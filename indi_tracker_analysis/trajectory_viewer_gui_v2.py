from optparse import OptionParser
import sys, os
import imp

import rosbag, rospy
import pickle

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.ptime as ptime
import time
import numpy as np

from multi_tracker_analysis import read_hdf5_file_to_pandas
from multi_tracker_analysis import data_slicing
import indi_tracker_analysis.find_flies_in_image_directory as find_flies_in_image_directory
from indi_tracker_analysis.find_flies_in_image_directory import FlyImg

import calibrate_gphoto2_camera
import split_bag

import matplotlib.pyplot as plt

import multi_tracker_analysis as mta

import cv2
import copy

import progressbar

import pandas

import subprocess
import warnings

from distutils.version import LooseVersion, StrictVersion

print('Using numpy: ' + np.version.version)
print('Using pyqtgraph: ' + pg.__version__)

# video would not load before installing most recent version of pyqtgraph from github repo
# this is the version of the commit that fixed the
# issue with current numpy: pyqtgraph-0.9.10-118-ge495bbc (in commit e495bbc...)
# version checking with distutils.version. See: http://stackoverflow.com/questions/11887762/compare-version-strings
if StrictVersion(pg.__version__) < StrictVersion("0.9.10"):
    if StrictVersion(np.version.version) > StrictVersion("1.10"):
        warnings.warn('Using pyqtgraph may be incompatible with numpy. Video may not load.')
        quit()
pg.mkQApp()

try:
	import gtk, pygtk
except:
	from gi.repository import Gtk as gtk
window = gtk.Window()
screen = window.get_screen()
screen_width = screen.get_width()
screen_height = screen.get_height()

# check screen size - if "small" screen use smaller ui
path = os.path.dirname(os.path.abspath(__file__))
if screen_height < 4000:
    #uiFile = os.path.join(path, 'trajectory_viewer_small_screens.ui')
    uiFile = os.path.join(path, 'trajectory_viewer_gui_gphoto2_2700.ui')
    SMALL = True
else:
    uiFile = os.path.join(path, 'trajectory_viewer_gui_gphoto2.ui')
    SMALL = False
#uiFile = '/home/caveman/catkin_ws/src/multi_tracker/multi_tracker_analysis/trajectory_viewer_small_screens.ui'
WindowTemplate, TemplateBaseClass = pg.Qt.loadUiType(uiFile)

def get_random_color():
    color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
    return color

def convert_identifier_code_to_epoch_time(identifiercode):
    ymd, hms, n = identifiercode.split('_')
    strt = ymd + '_' + hms
    t = time.strptime(strt, "%Y%m%d_%H%M%S")
    epoch_time = time.mktime(t) # assuming PDT / PST
    return epoch_time

  
class QTrajectory(TemplateBaseClass):
    def __init__(self, data_filename, bgimg, delta_video_filename, load_original=False, clickable_width=6, draw_interesting_time_points=True, draw_config_function=False, chunked_dvbag=False):
        self.load_original = load_original 
        
        TemplateBaseClass.__init__(self)
        self.setWindowTitle('Trajectory Viewer GUI v2')
    
        # Create the main window
        #self.app = QtGui.QApplication([])
        self.ui = WindowTemplate()
        self.ui.setupUi(self)
        #self.show()

        # options
        self.draw_interesting_time_points = draw_interesting_time_points
        self.draw_config_function = draw_config_function
        self.chunked_dvbag = chunked_dvbag

        # Buttons
        self.ui.save_trajecs.clicked.connect(self.save_trajectories)
        self.ui.movie_save.clicked.connect(self.save_image_sequence)
        self.ui.movie_speed.sliderMoved.connect(self.set_movie_speed)
        self.ui.trajec_undo.clicked.connect(self.trajec_undo)
        self.ui.movie_play.clicked.connect(self.movie_play)
        self.ui.movie_pause.clicked.connect(self.movie_pause)
        self.ui.trajec_delete.clicked.connect(self.toggle_trajec_delete)
        self.ui.trajec_cut.clicked.connect(self.toggle_trajec_cut)
        self.ui.trajec_join_collect.clicked.connect(self.toggle_trajec_join_collect)
        self.ui.trajec_select_all.clicked.connect(self.select_all_trajecs)
        self.ui.trajec_select_drag.clicked.connect(self.toggle_trajec_drag)
        self.ui.trajec_join_add_data.clicked.connect(self.toggle_trajec_join_add_data)
        self.ui.trajec_join_save.clicked.connect(self.trajec_join_save)
        self.ui.trajec_join_clear.clicked.connect(self.toggle_trajec_join_clear)
        self.ui.get_original_objid.clicked.connect(self.trajec_get_original_objid)
        self.ui.save_annotation.clicked.connect(self.save_annotation)
        self.ui.load_annotations.clicked.connect(self.load_annotations)
        self.ui.annotated_color_checkbox.stateChanged.connect(self.toggle_annotated_colors)
        self.ui.annotated_hide_checkbox.stateChanged.connect(self.toggle_annotated_hide)
        self.ui.save_colors.clicked.connect(self.save_trajec_colors)


        self.ui.get_original_objid.clicked.connect(self.trajec_get_original_objid)
        self.ui.save_colors.clicked.connect(self.save_trajec_colors)

        self.ui.selection_radius.setPlainText(str(100))
        self.ui.min_selection_length.setPlainText(str(0))
        self.ui.max_selection_length.setPlainText(str(-1)) # -1 means all
        
        # parameters
        self.data_filename = data_filename
        self.load_data()
        self.backgroundimg_filename = bgimg
        self.backgroundimg = None
        self.binsx = None
        self.binsy = None
        trange = np.float( np.max(self.pd.time_epoch.values) - np.min(self.pd.time_epoch.values) )
        self.troi = [np.min(self.pd.time_epoch.values), np.min(self.pd.time_epoch.values)+trange*0.1] 
        self.skip_frames = 1
        self.frame_delay = 0.03
        self.path = os.path.dirname(data_filename)
        self.clickable_width = clickable_width
        
        # load delta video bag
        if not self.chunked_dvbag:
            print('Loading delta video')
            if delta_video_filename != 'none':
                self.dvbag = rosbag.Bag(delta_video_filename)
            else:
                self.dvbag = None
            print('Done loading delta video')
        else:
            print('Chunked bag directory: ')
            self.delta_video_dirname = delta_video_filename.split('.bag')[0] + "_chunked"
            print(self.delta_video_dirname)
            self.delta_video_time_to_chunk_dict = pandas.read_hdf(os.path.join(self.delta_video_dirname, "time_to_chunk_dict.hdf"))
            print('First chunk: ')
            print(self.delta_video_time_to_chunk_dict.chunkname.values[0])

        # find gphoto2 directory
        s = self.config.identifiercode + '_' + 'gphoto2'
        self.gphoto2directory = os.path.join(self.config.path, s)
        if os.path.exists( self.gphoto2directory ):
            self.draw_gphoto2_timepoints()
        else:
            self.gphoto2directory = None 
            
        # Initialize 
        self.trajec_width_dict = {}
        try:
            fname = os.path.join(self.path, 'trajec_to_color_dict.pickle')
            f = open(fname, 'r+')
            self.trajec_to_color_dict = pickle.load(f)
            f.close()
        except:
            self.trajec_to_color_dict = {}
            for key in self.pd.objid.unique():
                color = get_random_color()
                self.trajec_to_color_dict.setdefault(key, color)
        self.plotted_traces_keys = []
        self.plotted_traces = []
        self.trajectory_ends_vlines = []
        self.data_to_add = []
        self.selected_trajectory_ends = []
        self.object_id_numbers = []
        
        self.annotations = os.path.join(self.path, 'annotations.pickle')
        if os.path.exists(self.annotations):
            f = open(self.annotations, 'r+')
            data = pickle.load(f)
            f.close()
            self.annotated_keys = data.keys()
        else:
            self.annotated_keys = []
        
        self.time_mouse_click = time.time()
        self.cut_objects = False
        self.delete_objects = False
        self.join_objects = False
        self.add_data = False
        self.drag = False

        self.selection_radius = self.ui.__getattribute__('selection_radius')
        self.selection_radius = int(self.selection_radius.toPlainText())
        self.drag_rect = {'center_x': 100, 'center_y': 100, 'w': self.selection_radius, 'h': self.selection_radius}
        
        self.crosshair_pen = pg.mkPen('w', width=1)
        
        self.ui.qtplot_timetrace.enableAutoRange('xy', False)
        if self.config is not None:
            print('**** Sensory stimulus: ', self.config.sensory_stimulus_on)
            for r, row in enumerate(self.config.sensory_stimulus_on):
                v1 = pg.PlotDataItem([self.config.sensory_stimulus_on[r][0],self.config.sensory_stimulus_on[r][0]], [0,10])
                v2 = pg.PlotDataItem([self.config.sensory_stimulus_on[r][-1],self.config.sensory_stimulus_on[r][-1]], [0,10])
                try:
                    f12 = pg.FillBetweenItem(curve1=v1, curve2=v2, brush=pg.mkBrush(self.config.sensory_stimulus_rgba[r]) )
                except:
                    f12 = pg.FillBetweenItem(curve1=v1, curve2=v2, brush=pg.mkBrush((255,0,0,150)) )
                self.ui.qtplot_gphoto2times.addItem(f12)
        
        lr = pg.LinearRegionItem(values=self.troi)
        f = 'update_time_region'
        lr.sigRegionChanged.connect(self.__getattribute__(f))
        self.ui.qtplot_timetrace.addItem(lr)
        
        print('drawing interesting time points')
        self.draw_timeseries_vlines_for_interesting_timepoints()
        print('done drawing interesting time points')
        self.ui.qtplot_timetrace.setRange(xRange=[np.min(self.time_epoch_continuous), np.max(self.time_epoch_continuous)], yRange=[0, np.max(self.nflies)])
        self.ui.qtplot_timetrace.setLimits(yMin=0, yMax=np.max(self.nflies))
        self.ui.qtplot_timetrace.setLimits(minYRange=np.max(self.nflies), maxYRange=np.max(self.nflies))
        
        self.current_time_vline = pg.InfiniteLine(angle=90, movable=False)
        self.ui.qtplot_timetrace.addItem(self.current_time_vline, ignoreBounds=True)
        self.current_time_vline.setPos(0)
        pen = pg.mkPen((255,255,255), width=2)
        self.current_time_vline.setPen(pen)

        self.ui.qtplot_gphoto2times.enableAutoRange('xy', False)
        self.ui.qtplot_gphoto2times.setRange(xRange=[np.min(self.time_epoch_continuous), np.max(self.time_epoch_continuous)], yRange=[0, 1])
        self.ui.qtplot_gphoto2times.setLimits(yMin=0, yMax=1)
        self.ui.qtplot_gphoto2times.setLimits(minYRange=1, maxYRange=1)

        #self.ui.qtplot_timetrace.sigXRangeChanged.connect(self.update_time_range_gphoto2)
        #self.ui.qtplot_gphoto2times.sigXRangeChanged.connect(self.update_time_range_timetrace)
        self.ui.qtplot_gphoto2times.setXLink(self.ui.qtplot_timetrace)

        # hide a bunch of the axes
        self.ui.qtplot_gphoto2times.hideAxis('left')
        self.ui.qtplot_gphoto2times.hideAxis('bottom')

        self.ui.qtplot_timetrace.hideAxis('left')
        self.ui.qtplot_timetrace.hideAxis('bottom')

        self.ui.qtplot_trajectory.hideAxis('left')
        self.ui.qtplot_trajectory.hideAxis('bottom')

        self.ui.qtplot_gphoto2image.hideAxis('left')
        self.ui.qtplot_gphoto2image.hideAxis('bottom')

        self.fly_letters = ['A', 'B', 'C', 'D', 'E', 'F']
        img_boxes = [eval('self.ui.qtplot_gphoto2_fly' + a + '_img') for a in self.fly_letters]
        for i in range(len(img_boxes)):
            img_boxes[i].hideAxis('left')
            img_boxes[i].hideAxis('bottom')
        #

    ### Button Callbacks

    def save_trajectories(self):
        self.troi = self.linear_region.getRegion()

        start_frame = self.dataset.timestamp_to_framestamp(self.troi[0])
        end_frame = self.dataset.timestamp_to_framestamp(self.troi[-1])
        dirname = 'data_selection_' + str(start_frame) + '_to_' + str(end_frame)
        dirname = os.path.join(self.path, dirname)

        if os.path.exists(dirname):
            print 'Data selection path exists!'
        else:
            os.mkdir(dirname)

        fname = 'dataframe_' + str(start_frame) + '_to_' + str(end_frame) + '.pickle'
        fname = os.path.join(dirname, fname)
        print 'Saving stand alone pandas dataframe to file: '
        print '    ' + fname

        pd_subset = mta.data_slicing.get_data_in_epoch_timerange(self.pd, self.troi)

        pd_subset.to_pickle(fname)

        #self.config.plot_trajectories(self.troi)
    
    def set_all_buttons_false(self):
        self.cut_objects = False
        self.join_objects = False
        self.delete_objects = False
        self.add_data = False
        self.get_original_objid = False
        self.drag = False
        self.draw_trajectories()
    
    def set_movie_speed(self, data):
        if data >0:
            self.skip_frames = data
            self.frame_Delay = 0.03
        if data == 0:
            self.skip_frames = 1
            self.frame_delay = 0.03
        if data <0:
            p = 1- (np.abs(data) / 30.)
            max_frame_delay = 0.2
            self.frame_delay = (max_frame_delay - (max_frame_delay*p))*2
            
    def get_annotations_from_checked_boxes(self):
        notes = []
        for i in range(1,5):
            checkbox = self.ui.__getattribute__('annotated_checkbox_' + str(i))
            if checkbox.checkState():
                textbox = self.ui.__getattribute__('annotated_text_' + str(i))
                note = textbox.toPlainText()
                notes.append(str(note))
        return notes
    
    def save_annotation(self, key=None):
        if key == None or key == False:
            keys_to_annotate = self.object_id_numbers
        else:
            keys_to_annotate = [key]

        print 'Saving annotation for: ', keys_to_annotate
        notes = self.get_annotations_from_checked_boxes()
        notes = list(set(notes)) # make sure all notes are unique

        print(notes)
        self.annotations = os.path.join(self.path, 'annotations.pickle')
        if os.path.exists(self.annotations):
            f = open(self.annotations, 'r+')
            data = pickle.load(f)
            f.close()
        else:
            f = open(self.annotations, 'w+')
            f.close()
            data = {}


        
        for key in keys_to_annotate:
            if key not in data.keys():
                data.setdefault(key, {'notes': [], 'related_objids': []})
            data[key]['notes'] = notes
            data[key]['related_objids'] = self.object_id_numbers
            if len(notes) == 0:
                del(data[key])
        self.annotated_keys = data.keys()

        f = open(self.annotations, 'r+')
        pickle.dump(data, f)
        f.close()
        print('Saved annotation')
        
        self.toggle_trajec_join_clear()
                
    def load_annotations(self):
        for i in range(1,5):
            checkbox = self.ui.__getattribute__('annotated_checkbox_' + str(i))
            checkbox.setCheckState(0)
            textbox = self.ui.__getattribute__('annotated_text_' + str(i))
            textbox.clear()
            
        self.annotations = os.path.join(self.path, 'annotations.pickle')
        if os.path.exists(self.annotations):
            f = open(self.annotations, 'r+')
            data = pickle.load(f)
            f.close()
        else:
            data = {}
            
        notes = []
        for key in self.object_id_numbers:
            if key in data.keys():
                annotation = data[key]
                notes.extend(annotation['notes'])

        if len(notes) > 0:
            for i, note in enumerate(notes):
                checkbox = self.ui.__getattribute__('annotated_checkbox_' + str(i+1))
                checkbox.setChecked(True)
                textbox = self.ui.__getattribute__('annotated_text_' + str(i+1))
                textbox.setPlainText(note)
        
    def toggle_annotated_colors(self):
        self.draw_trajectories()
    
    def toggle_annotated_hide(self):
        self.draw_trajectories()
        
    def save_trajec_colors(self):
        fname = os.path.join(self.path, 'trajec_to_color_dict.pickle')
        f = open(fname, 'w+')
        pickle.dump(self.trajec_to_color_dict, f)
        f.close()
        
    def trajec_undo(self):
        self.toggle_trajec_join_clear()
        instruction = self.instructions.pop(-1)
        filename = os.path.join(self.path, 'delete_cut_join_instructions.pickle')
        if os.path.exists(filename):
            f = open(filename, 'r+')
            data = pickle.load(f)
            f.close()
        else:
            f = open(filename, 'w+')
            f.close()
            data = []
        data = self.instructions
        f = open(filename, 'r+')
        pickle.dump(data, f)
        f.close()
        time.sleep(1)
        self.load_data()
        self.draw_trajectories()
        self.draw_timeseries_vlines_for_interesting_timepoints()
        
    def movie_pause(self):
        if self.play is True:
            self.play = False
            print('pause movie')
        elif self.play is False:
            self.play = True
            print('playing movie')
            self.updateTime = ptime.time()
            self.updateData()
            
    def movie_play(self):
        self.play = True
        print('loading image sequence')
        self.load_image_sequence()
    
        print('playing movie')
        self.updateTime = ptime.time()
        self.updateData()
        
    def trajec_get_original_objid(self):
        self.set_all_buttons_false()
        self.get_original_objid = True
        self.crosshair_pen = pg.mkPen((255, 129, 234), width=1)
    
    def toggle_trajec_delete(self):
        self.set_all_buttons_false()
        self.delete_objects = True
        self.crosshair_pen = pg.mkPen('r', width=1)
        print('Deleting objects!')

        # first delete selected trajectories
        
        if len(self.object_id_numbers) == 1:
            print 'Deleting selected objects: ', self.object_id_numbers
            while len(self.object_id_numbers) > 0:
                key = self.object_id_numbers.pop()
            self.delete_object_id_number(key, redraw=False)
        else:
            print 'Mass deleting ' + str(len(self.object_id_numbers)) + ' objects'
            self.delete_object_id_group(self.object_id_numbers, redraw=False)
            self.object_id_numbers = []

        self.draw_trajectories()
        self.draw_timeseries_vlines_for_interesting_timepoints()
        #
    
    def toggle_trajec_cut(self):
        self.set_all_buttons_false()

        self.cut_objects = True
        self.crosshair_pen = pg.mkPen('y', width=1)
        print('Cutting objects!')
    
    def toggle_trajec_join_collect(self):
        self.set_all_buttons_false()
        
        self.join_objects = True
        self.crosshair_pen = pg.mkPen('g', width=1)
        self.ui.qttext_selected_objids.clear()
        
        print('Ready to collect object id numbers. Click on traces to add object id numbers to the list. Click "save object id numbers" to save, and reset the list')
    
    def toggle_trajec_drag(self):
        self.set_all_buttons_false()
        
        self.drag = True
        self.selection_radius = self.ui.__getattribute__('selection_radius')
        self.selection_radius = int(self.selection_radius.toPlainText())
        self.drag_rect['w'] = self.selection_radius
        self.drag_rect['h'] = self.selection_radius
        self.crosshair_pen = pg.mkPen('m', width=1)
        self.ui.qttext_selected_objids.clear()

        self.toggle_trajec_join_clear(join=False)

        #self.draw_trajectories()
        
        print('Click to place circle. Use text box to change radius and then click.')
    

    def select_all_trajecs(self):
        if not self.join_objects:
            #self.toggle_trajec_join_collect()
            self.join_objects = True
            self.ui.qttext_selected_objids.clear()
        
        if not SMALL:
            self.selection_radius = self.ui.__getattribute__('selection_radius')
            self.selection_radius = int(self.selection_radius.toPlainText())

            min_len = self.ui.__getattribute__('min_selection_length')
            min_len = int(min_len.toPlainText())

            max_len = self.ui.__getattribute__('max_selection_length')
            max_len = int(max_len.toPlainText())
            if max_len == -1:
                max_len = np.inf
        else:
            min_len = 0
            max_len = np.inf
            
        pd_subset = mta.data_slicing.get_data_in_epoch_timerange(self.pd, self.troi)
        keys = np.unique(pd_subset.objid.values)

        print('There are ' + str(len(self.plotted_traces)) + ' traces plotted')
        for key in keys:
            if len(self.plotted_traces) > 0:
                trace = self.plotted_traces[self.plotted_traces_keys.index(key)]
                
            #key = trace.curve.key
            trajec_length = len(self.pd[self.pd.objid==key])
            if trajec_length > min_len and trajec_length < max_len:
                if not self.drag:
                    self.trace_clicked(trace.curve)
                    self.object_id_numbers.append(key)
                else:
                    # check to see if trajec in radius
                    trajec = self.dataset.trajec(key)
                    first_time = np.max([self.troi[0], trajec.time_epoch[0]])
                    first_time_index = np.argmin( np.abs(trajec.time_epoch-first_time) )
                    last_time = np.min([self.troi[-1], trajec.time_epoch[-1]])
                    last_time_index = np.argmin( np.abs(trajec.time_epoch-last_time) )

                    trajec_y = trajec.position_y[first_time_index:last_time_index]
                    trajec_x = trajec.position_x[first_time_index:last_time_index]
                    
                    trajec_dist = np.sqrt((trajec_y - self.drag_rect['center_x'])**2 + (trajec_x - self.drag_rect['center_y'])**2 )

                    if self.selection_radius > 0:
                        if (trajec_dist < self.selection_radius/2.).any():
                            if len(self.plotted_traces) > 0:
                                self.trace_clicked(trace.curve)
                            else:
                                self.object_id_numbers.append(key)
                    else:
                        if (trajec_dist > np.abs(self.selection_radius)/2.).all():

                            if len(self.plotted_traces) > 0:
                                self.trace_clicked(trace.curve)
                            else:
                                self.object_id_numbers.append(key)

        self.object_id_numbers = list(np.unique(self.object_id_numbers))

        self.join_objects = False
        self.ui.qttext_selected_objids.setPlainText(str(self.object_id_numbers))
        #self.ui.qttext_selected_objids.clear()

    def toggle_trajec_join_add_data(self):
        self.set_all_buttons_false()

        self.data_to_add = []
        self.add_data = True
        self.crosshair_pen = pg.mkPen((0,0,255), width=1)
        print('Adding data!')
   
    def toggle_trajec_join_clear(self, join=True):
        if join:
            self.set_all_buttons_false()

        self.trajec_width_dict = {}

        for key in self.object_id_numbers:
            self.trajec_width_dict[key] = 2

        self.crosshair_pen = pg.mkPen('w', width=1)
        self.object_id_numbers = []
        self.add_data = []
        self.ui.qttext_selected_objids.clear()
        print('Join list cleared')


        self.draw_trajectories()
        
        if join:
            self.toggle_trajec_join_collect()
    
    ### Mouse moved / clicked callbacks
    
    def mouse_moved(self, pos):
        self.mouse_position = [self.img.mapFromScene(pos).x(), self.img.mapFromScene(pos).y()]
        self.crosshair_vLine.setPos(self.mouse_position[0])
        self.crosshair_hLine.setPos(self.mouse_position[1])
            
        self.crosshair_vLine.setPen(self.crosshair_pen)
        self.crosshair_hLine.setPen(self.crosshair_pen)

        #print(self.mouse_position)
        
    def mouse_clicked(self, data):
        print('click')
        self.time_since_mouse_click = time.time() - self.time_mouse_click
        if self.time_since_mouse_click > 0.5:
            if self.add_data:
                self.add_data_to_trajecs_to_join()
        self.time_mouse_click = time.time()
        
        if self.drag:
            if self.mouse_position[0] != self.drag_rect['center_x']:
                self.selection_radius = self.ui.__getattribute__('selection_radius')
                self.selection_radius = int(self.selection_radius.toPlainText())

                self.drag_rect['h'] = self.selection_radius
                self.drag_rect['w'] = self.selection_radius
                self.drag_rect['center_x'] = self.mouse_position[0]
                self.drag_rect['center_y'] = self.mouse_position[1]


                self.toggle_trajec_join_clear(join=False)

                self.draw_trajectories()

        try:
            if self.get_original_objid:
                s = 'time_epoch > ' + str(self.current_time_epoch - 1) + ' & time_epoch < ' + str(self.current_time_epoch + 1)
                pd_tmp = self.original_pd.query(s)
                print s, pd_tmp.shape
                x_diff = np.abs(pd_tmp.position_x.values - self.mouse_position[1])
                y_diff = np.abs(pd_tmp.position_y.values - self.mouse_position[0])
                i = np.argmin(x_diff + y_diff)
                objid = pd_tmp.iloc[i].objid
                self.ui.qttext_show_original_objid.clear()
                self.ui.qttext_show_original_objid.setPlainText(str(int(objid)))
        except:
            pass

    def trace_clicked(self, item): 
        if self.join_objects:
            if item.key not in self.object_id_numbers:
                print 'Saving object to object list: ', item.key
                self.object_id_numbers.append(item.key)
                color = self.trajec_to_color_dict[item.key]
                pen = pg.mkPen(color, width=4)  
                self.trajec_width_dict.setdefault(item.key, 4)
                item.setPen(pen)
            else:
                print 'Removing object from object list: ', item.key
                self.object_id_numbers.remove(item.key)
                color = self.trajec_to_color_dict[item.key]
                pen = pg.mkPen(color, width=2)  
                self.trajec_width_dict.setdefault(item.key, 2)
                item.setPen(pen)
            self.ui.qttext_selected_objids.clear()
            self.ui.qttext_selected_objids.setPlainText(str(self.object_id_numbers))
            self.draw_vlines_for_selected_trajecs()
            
        elif self.cut_objects:
            print 'Cutting trajectory: ', item.key, ' at: ', self.mouse_position
            self.cut_trajectory(item.key, self.mouse_position)
        elif self.delete_objects:
            self.delete_object_id_number(item.key)
        elif self.add_data:
            self.add_data_to_trajecs_to_join()

        self.load_annotations()
            
    def add_data_to_trajecs_to_join(self):
        self.data_to_add.append([self.current_time_epoch, self.mouse_position[0], self.mouse_position[1]])
        self.draw_data_to_add()
        
    def get_new_unique_objid(self):
        fname = os.path.join(self.path, 'new_unique_objids.pickle')
        if os.path.exists(fname):
            f = open(fname, 'r+')
            data = pickle.load(f)
            f.close()
        else:
            f = open(fname, 'w+')
            f.close()
            data = [np.max(self.pd.objid)+10]
        new_objid = data[-1] + 1
        data.append(new_objid)
        f = open(fname, 'r+')
        pickle.dump(data, f)
        f.close()
        print 'NEW OBJID CREATED: ', new_objid
        return new_objid
        
    def cut_trajectory(self, key, point):
        dataset = mta.read_hdf5_file_to_pandas.Dataset(self.pd)
        trajec = dataset.trajec(key)
        p = np.vstack((trajec.position_y, trajec.position_x))
        point = np.array([[point[0]], [point[1]]])
        error = np.linalg.norm(p-point, axis=0)
        trajectory_frame = np.argmin(error)
        dataset_frame = dataset.timestamp_to_framestamp(trajec.time_epoch[trajectory_frame])
        
        instructions = {'action': 'cut',
                        'order': time.time(),
                        'objid': key,
                        'cut_frame_global': dataset_frame,
                        'cut_frame_trajectory': trajectory_frame, 
                        'cut_time_epoch': trajec.time_epoch[trajectory_frame],
                        'new_objid': self.get_new_unique_objid()}
        self.save_delete_cut_join_instructions(instructions)

        # update gui
        self.pd = mta.read_hdf5_file_to_pandas.delete_cut_join_trajectories_according_to_instructions(self.pd, instructions, interpolate_joined_trajectories=True)
        self.draw_trajectories(cut=True)
        self.draw_timeseries_vlines_for_interesting_timepoints()
        
    def trajec_join_save(self):
        instructions = {'action': 'join',
                        'order': time.time(),
                        'objids': self.object_id_numbers,
                        'data_to_add': self.data_to_add,
                        'new_objid': self.get_new_unique_objid()}
        print instructions

        # copy over annotations
        self.load_annotations()
        self.save_annotation(instructions['new_objid'])

        self.save_delete_cut_join_instructions(instructions)
        
        self.object_id_numbers = []
        self.ui.qttext_selected_objids.clear()
        self.data_to_add = []
        self.trajec_width_dict = {}
        
        # now join them for the gui
        self.pd = mta.read_hdf5_file_to_pandas.delete_cut_join_trajectories_according_to_instructions(self.pd, instructions, interpolate_joined_trajectories=True)
        self.draw_trajectories()
        self.draw_timeseries_vlines_for_interesting_timepoints()


        
        print 'Reset object id list - you may collect a new selection of objects now'
        
    def delete_object_id_group(self, keys, redraw=True):
        instructions = {'action': 'delete',
                        'order': time.time(),
                        'objid': keys}
        self.save_delete_cut_join_instructions(instructions)
        # update gui
        #self.trajec_to_color_dict[key] = (0,0,0,0) 
        self.pd = mta.read_hdf5_file_to_pandas.delete_cut_join_trajectories_according_to_instructions(self.pd, instructions, interpolate_joined_trajectories=True)
        if redraw:
            self.draw_trajectories()
            self.draw_timeseries_vlines_for_interesting_timepoints()

    def delete_object_id_number(self, key, redraw=True):
        instructions = {'action': 'delete',
                        'order': time.time(),
                        'objid': key}
        self.save_delete_cut_join_instructions(instructions)
        # update gui
        #self.trajec_to_color_dict[key] = (0,0,0,0) 
        self.pd = mta.read_hdf5_file_to_pandas.delete_cut_join_trajectories_according_to_instructions(self.pd, instructions, interpolate_joined_trajectories=True)
        if redraw:
            self.draw_trajectories()
            self.draw_timeseries_vlines_for_interesting_timepoints()
    
    ### Drawing functions
    
    def draw_timeseries_vlines_for_interesting_timepoints(self):
        if 1:
            self.calc_time_etc()
            
            # clear
            try:
                self.ui.qtplot_timetrace.removeItem(self.nflies_plot)
            except:
                pass
            for vline in self.trajectory_ends_vlines:
                self.ui.qtplot_timetrace.removeItem(vline)
            self.trajectory_ends_vlines = []
            
            # draw
            self.nflies_plot = self.ui.qtplot_timetrace.plot(x=self.time_epoch_continuous, y=self.nflies)
            
        if self.draw_interesting_time_points:
            objid_ends = self.pd.groupby('objid').time_epoch.max()
            for key in objid_ends.keys():
                t = objid_ends[key]
                vline = pg.InfiniteLine(angle=90, movable=False)
                self.ui.qtplot_timetrace.addItem(vline, ignoreBounds=True)
                vline.setPos(t)
                pen = pg.mkPen(self.trajec_to_color_dict[key], width=1)
                vline.setPen(pen)
                self.trajectory_ends_vlines.append(vline)
            
            # TODO: times (or frames) where trajectories get very close to one another
        
    def draw_vlines_for_selected_trajecs(self):
        for vline in self.selected_trajectory_ends:
            self.ui.qtplot_timetrace.removeItem(vline)
        self.selected_trajectory_ends = []
        for key in self.object_id_numbers:
            trajec = self.dataset.trajec(key)
            vline = pg.InfiniteLine(angle=90, movable=False)
            self.ui.qtplot_timetrace.addItem(vline, ignoreBounds=True)
            vline.setPos(trajec.time_epoch[-1])
            pen = pg.mkPen(self.trajec_to_color_dict[key], width=5)
            vline.setPen(pen)
            self.selected_trajectory_ends.append(vline)
            
    def draw_gphoto2_timepoints(self):

        if hasattr(self, 'gphoto2_calibration'):
            pass
        else:
            # note: use calibrate_gphoto2_camera.py to precalculate with optional delay
            try:
                self.gphoto2_calibration, self.gphoto2_delay_opt = calibrate_gphoto2_camera.get_optimal_gphoto2_homography(self.path) 
            except:
                print('Skipping calibration of gphoto images')
                self.gphoto2_calibration = None
                self.gphoto2_delay_opt = 0

        if hasattr(self, 'gphoto2_flies'):
            pass
        else:
            fname = os.path.join(self.path, 'flyimgs')
            if not os.path.exists(fname):
                raise ValueError('Please run find_flies_in_image_directory.extract_all_flyimgs')
            self.flyimg_dict = find_flies_in_image_directory.load_flyimg_dict_from_path(self.path)

        self.gphoto2_file_to_time = {}
        self.gphoto2_line_to_file = {}
        try:
            pens = self.gphoto2_pens
        except:
            self.gphoto2_pens = {}

        for filename, flyimg in self.flyimg_dict.items():
            s = flyimg.filename.split('_')
            time_epoch_secs = int(s[-2])
            time_epoch_nsecs = int(s[-1].split('.')[-2])
            time_epoch = float(time_epoch_secs) + float(time_epoch_nsecs*1e-9) - self.gphoto2_delay_opt

            flies = flyimg.fly_ellipses_small
            if len(flies) == 0:
                self.gphoto2_pens[flyimg.filename] = pg.mkPen(0.2, width=2)

            if flyimg.filename in self.gphoto2_pens.keys():
                pen = self.gphoto2_pens[flyimg.filename]
            else:
                pen = pg.mkPen(1, width=5)
                self.gphoto2_pens[flyimg.filename] = pen
            pline = self.ui.qtplot_gphoto2times.plot([time_epoch, time_epoch+0.0001], [0,1], pen=pen) 
            pline.curve.setClickable(True, width=self.clickable_width)
            pline.curve.filename = flyimg.filename
            pline.curve.sigClicked.connect(self.gphoto2_clicked)

            self.gphoto2_file_to_time[flyimg.filename] = time_epoch
            self.gphoto2_line_to_file[pline] = flyimg.filename

    def gphoto2_clicked(self, item):
        print item.filename
        for pline, file in self.gphoto2_line_to_file.items():
            if file == item.filename:
                pen = pg.mkPen((50,50,255), width=10)
            else:
                pen = self.gphoto2_pens[os.path.basename(file)]
            pline.setPen(pen)

        gphoto2_path = os.path.join(self.path, self.config.identifiercode + '_gphoto2')
        complete_filename = os.path.join(gphoto2_path, item.filename)
        gphoto2img = cv2.cvtColor(cv2.imread(complete_filename), cv2.COLOR_BGR2RGB) 
        
        img_boxes = [eval('self.ui.qtplot_gphoto2_fly' + a + '_img') for a in self.fly_letters]
        last_fly = -1
        self.gphoto2_fly_ellipses_to_draw_on_tracker = []


        flyimg = self.flyimg_dict[item.filename]
        for i in range(len(flyimg.fly_ellipses_small)):
            rgb_color = get_random_color()

            ellipse = flyimg.fly_ellipses_small[i]
            ellipse_large = flyimg.fly_ellipses_large[i]

            if self.gphoto2_calibration is not None:
                tracker_point = calibrate_gphoto2_camera.reproject_gphoto2_point_onto_tracker([ellipse[0]], self.gphoto2_calibration)[0]
                fly = {'tracker_point': tracker_point, 'color': rgb_color}
                self.gphoto2_fly_ellipses_to_draw_on_tracker.append(fly)

                # ellipse mask
                enlarged_ellipse_large = (( ellipse_large[0][0], ellipse_large[0][1]),
                                                (int(ellipse_large[1][0]*2), int(ellipse_large[1][1]*2)),
                                                ellipse_large[2])

                gphoto2img = cv2.ellipse(gphoto2img,enlarged_ellipse_large,rgb_color,5)
                last_fly = i

                if i < len(img_boxes):
                    zoom = flyimg.convert_bgr_to_rgb(flyimg.rois_fly[i]) 
                    try:
                        actual_width = np.max(zoom.shape) # won't work for corners
                        ellipse_large_centered = (( int(actual_width/2.), int(actual_width/2.)),
                                                    (int(enlarged_ellipse_large[1][0]), int(enlarged_ellipse_large[1][1])),
                                                    enlarged_ellipse_large[2])
                        zoom = cv2.ellipse(zoom,ellipse_large_centered,rgb_color,5)

                        zoom = pg.ImageItem(zoom, autoLevels=False)
                        img_boxes[i].clear()
                        img_boxes[i].addItem(zoom)
                    except:
                        print('Failed to get zoomed fly')
                        pass
                    
        for j in range(last_fly+1, len(img_boxes)):
            img_boxes[j].clear()

        gphoto2img = pg.ImageItem(gphoto2img, autoLevels=False)
        self.ui.qtplot_gphoto2image.clear()
        self.ui.qtplot_gphoto2image.addItem(gphoto2img)

        self.draw_trajectories()

    def update_time_region(self, linear_region):
        self.linear_region = linear_region
        self.troi = linear_region.getRegion()
        self.draw_trajectories()

    def init_bg_image(self):
        if self.binsx is None:
            self.binsx, self.binsy = mta.plot.get_bins_from_backgroundimage(self.backgroundimg_filename)
            self.backgroundimg = cv2.imread(self.backgroundimg_filename, cv2.CV_8UC1)
        img = copy.copy(self.backgroundimg)

        self.img = pg.ImageItem(img, autoLevels=False)

        
    def draw_trajectories(self, cut=False):
        tstart = self.start_time_epoch
        self.ui.qttext_time_range.setPlainText(str(self.troi[0]) + '\nto\n' + str(self.troi[-1]) ) 

        for plotted_trace in self.plotted_traces:
            try:
                self.ui.qtplot_trajectory.removeItem(plotted_trace)
            except:
                pass # item probably does not exist anymore
        self.ui.qtplot_trajectory.clear()

        
        
        pd_subset = mta.data_slicing.get_data_in_epoch_timerange(self.pd, self.troi)
        self.dataset = read_hdf5_file_to_pandas.Dataset(self.pd)
        
        self.init_bg_image()


        
        # plot a heatmap of the trajectories, for error checking
        h = mta.plot.get_heatmap(pd_subset, self.binsy, self.binsx, position_x='position_y', position_y='position_x', position_z='position_z', position_z_slice=None)
        indices = np.where(h != 0)

        img = copy.copy(self.backgroundimg)
        img[indices] = 0

        self.img = pg.ImageItem(img, autoLevels=False)
        self.ui.qtplot_trajectory.addItem(self.img)
        self.img.setZValue(-200)  # make sure image is behind other data
        
        # drag box
        if self.drag:
            self.draw_drag_rect()

        # cross hair mouse stuff
        self.ui.qtplot_trajectory.scene().sigMouseMoved.connect(self.mouse_moved)
        self.ui.qtplot_trajectory.scene().sigMouseClicked.connect(self.mouse_clicked)
        self.crosshair_vLine = pg.InfiniteLine(angle=90, movable=False)
        self.crosshair_hLine = pg.InfiniteLine(angle=0, movable=False)
        self.ui.qtplot_trajectory.addItem(self.crosshair_vLine, ignoreBounds=True)
        self.ui.qtplot_trajectory.addItem(self.crosshair_hLine, ignoreBounds=True)

        #if cut:
        #    return

        keys = np.unique(pd_subset.objid.values)
        try:
            _ = self.old_plotted_traces
        except:
            self.old_plotted_traces = []
        try:
            _ = self.plotted_traces
        except:
            _ = None
        if _ is not None:
            for curve in self.plotted_traces:
                self.ui.qtplot_trajectory.removeItem(curve)
                self.old_plotted_traces.append(curve) # this is horribly inefficient, but otherwise objects get deleted and that causes a crash

        self.plotted_traces = []
        self.plotted_traces_keys = []

        if len(keys) < 100:
            for key in keys:
                trajec = self.dataset.trajec(key)
                first_time = np.max([self.troi[0], trajec.time_epoch[0]])
                first_time_index = np.argmin( np.abs(trajec.time_epoch-first_time) )
                last_time = np.min([self.troi[-1], trajec.time_epoch[-1]])
                last_time_index = np.argmin( np.abs(trajec.time_epoch-last_time) )
                #if trajec.length > 5:
                if key not in self.trajec_to_color_dict.keys():
                    color = get_random_color()
                    self.trajec_to_color_dict.setdefault(key, color)
                else:
                    color = self.trajec_to_color_dict[key]
                if key in self.trajec_width_dict.keys():
                    width = self.trajec_width_dict[key]
                else:
                    width = 2
                if self.ui.annotated_color_checkbox.checkState():
                    if key in self.annotated_keys:
                        color = (0,0,0)
                        width = 6
                if self.ui.annotated_hide_checkbox.checkState():
                    if key in self.annotated_keys:
                        color = (0,0,0,0)
                        width = 1
                pen = pg.mkPen(color, width=width)  

                plotted_trace = self.ui.qtplot_trajectory.plot(trajec.position_y[first_time_index:last_time_index], trajec.position_x[first_time_index:last_time_index], pen=pen) 
                self.plotted_traces.append(plotted_trace)
                self.plotted_traces_keys.append(key)
                
            for i, key in enumerate(self.plotted_traces_keys):
                print key
                self.plotted_traces[i].curve.setClickable(True, width=self.clickable_width)
                self.plotted_traces[i].curve.key = key
                self.plotted_traces[i].curve.sigClicked.connect(self.trace_clicked)
        



        self.draw_data_to_add()
        self.draw_vlines_for_selected_trajecs()
        self.draw_gphoto2_flies_on_tracker()
        
        #self.save_trajec_color_width_dicts()

    def draw_drag_rect(self):
        w = self.drag_rect['w']
        h = self.drag_rect['h']
        x = self.drag_rect['center_x']-np.abs(int(w/2.))
        y = self.drag_rect['center_y']-np.abs(int(h/2.))

        r1 = pg.QtGui.QGraphicsEllipseItem(x, y, np.abs(w), np.abs(h))
        r1.setPen(pg.mkPen(None))
        if self.selection_radius > 0:
            r1.setBrush(pg.mkBrush(255,0,255,50))
        else:
            r1.setBrush(pg.mkBrush(0,255,0,50))

        self.ui.qtplot_trajectory.addItem(r1)

    def draw_data_to_add(self):
        for data in self.data_to_add:
            print data
            self.ui.qtplot_trajectory.plot([data[1]], [data[2]], pen=(0,0,0), symbol='o', symbolSize=10) 

    def draw_gphoto2_flies_on_tracker(self):
        if hasattr(self, 'gphoto2_fly_ellipses_to_draw_on_tracker'):
            for fly in self.gphoto2_fly_ellipses_to_draw_on_tracker:
                self.ui.qtplot_trajectory.plot([fly['tracker_point'][1]], [fly['tracker_point'][0]], symbolPen=pg.mkPen(color=fly['color'], symbol='o', symbolSize=10), symbolBrush=pg.mkBrush(color=fly['color']))#mkPen(color=fly['color']), symbol='o', symbolSize=10) 
    
    ### Load / read / save data functions
    
    def load_data(self):
        if self.load_original:
            self.original_pd = mta.read_hdf5_file_to_pandas.load_data_as_pandas_dataframe_from_hdf5_file(self.data_filename)
        print 'loading data'
        self.pd, self.config = mta.read_hdf5_file_to_pandas.load_and_preprocess_data(self.data_filename)
        self.start_time_epoch = convert_identifier_code_to_epoch_time(self.config.identifiercode)
        self.path = self.config.path
        self.dataset = read_hdf5_file_to_pandas.Dataset(self.pd)
        filename = os.path.join(self.path, 'delete_cut_join_instructions.pickle')
        if os.path.exists(filename):
            f = open(filename, 'r+')
            data = pickle.load(f)
            f.close()
        else:
            data = []
        self.instructions = data
        self.calc_time_etc()
        print 'data loaded'
        print 'N Trajecs: ', len(self.pd.groupby('objid'))
    
    def calc_time_etc(self):
        self.time_epoch = self.pd.time_epoch.groupby(self.pd.index).mean().values
        self.speed = self.pd.speed.groupby(self.pd.index).mean().values
        self.nflies = data_slicing.get_nkeys_per_frame(self.pd)
        if len(self.nflies) == 0:
            self.nflies = [0, 0, 0]
        try:
            self.time_epoch_continuous = np.linspace(np.min(self.time_epoch), np.max(self.time_epoch), len(self.nflies))
        except:
            self.time_epoch_continuous = np.linspace(0, 1, 1)
            
    def save_delete_cut_join_instructions(self, instructions):
        self.delete_cut_join_filename = os.path.join(self.path, 'delete_cut_join_instructions.pickle')
        if os.path.exists(self.delete_cut_join_filename):
            f = open(self.delete_cut_join_filename, 'r+')
            data = pickle.load(f)
            f.close()
        else:
            f = open(self.delete_cut_join_filename, 'w+')
            f.close()
            data = []
        data.append(instructions)
        f = open(self.delete_cut_join_filename, 'r+')
        pickle.dump(data, f)
        f.close()
        self.instructions.append(instructions)
  
    def load_image_sequence(self):
        try:
            del(self.image_sequence)
            del(self.msgs)
            for bag in self.bags:
                bag.close()
            print('Deleted loaded image sequence and closed bags')
        except:
            pass

        version = subprocess.check_output(["rosversion", "-d"])

        timerange = self.troi
        print 'loading image sequence from delta video bag - may take a moment'
        pbar = progressbar.ProgressBar().start()
        
        if not self.chunked_dvbag:
            rt0 = rospy.Time(timerange[0])
            rt1 = rospy.Time(timerange[1])
            self.msgs = self.dvbag.read_messages(start_time=rt0, end_time=rt1)
        else:
            print('Loading video from chunked bags')
            self.msgs, self.bags = split_bag.load_messages_for_chunked_bag(self.delta_video_time_to_chunk_dict, timerange[0], timerange[1])
            print('Done loading video from chunked bags')

        self.image_sequence = []
        self.image_sequence_timestamps = []
        t0 = None
        self.delta_video_background_img_filename = None
        self.delta_video_background_img = None
        
        for m, msg in enumerate(self.msgs):
            bag_time_stamp = float(msg[1].header.stamp.secs) + float(msg[1].header.stamp.nsecs)*1e-9
            delta_video_background_img_filename = os.path.join( self.path, os.path.basename(msg[1].background_image) )
            if os.path.exists(delta_video_background_img_filename):            
                if delta_video_background_img_filename != self.delta_video_background_img_filename:
                    self.delta_video_background_img_filename = delta_video_background_img_filename
                    self.delta_video_background_img = cv2.imread(self.delta_video_background_img_filename, cv2.CV_8UC1)
            else: # if we can't find the bgimg, do the best we can
                if self.delta_video_background_img is None:
                    self.delta_video_background_img_filename = mta.read_hdf5_file_to_pandas.get_filename(self.path, 'deltavideo_bgimg')
                    self.delta_video_background_img = cv2.imread(self.delta_video_background_img_filename, cv2.CV_8UC1)
                    
            imgcopy = copy.copy(self.delta_video_background_img)

            if len(msg[1].values) > 0:

                if 'kinetic' in version:
                    msg[1].xpixels = tuple(x - 1 for x in msg[1].xpixels)
                    msg[1].ypixels = tuple(y - 1 for y in msg[1].ypixels)
                else:
                    pass #print('Not ros kinetic.')

                imgcopy[msg[1].xpixels, msg[1].ypixels] = msg[1].values # if there's an error, check if you're using ROS hydro?
            
            if self.draw_config_function:
                imgcopy = cv2.cvtColor(imgcopy, cv2.COLOR_GRAY2RGB)
                self.config.draw(imgcopy, bag_time_stamp)

            self.image_sequence.append(imgcopy)
            #s = int((m / float(len(self.msgs)))*100)
            tfloat = msg[1].header.stamp.secs + msg[1].header.stamp.nsecs*1e-9
            self.image_sequence_timestamps.append(tfloat)
            if t0 is not None:
                t_elapsed = tfloat - t0
                t_total = timerange[1] - timerange[0]
                s = int(100*(t_elapsed / t_total))
                pbar.update(s)
            else:
                t0 = tfloat
        pbar.finish()
        self.current_frame = -1
        
    def save_image_sequence(self):
        start_frame = self.dataset.timestamp_to_framestamp(self.troi[0])
        end_frame = self.dataset.timestamp_to_framestamp(self.troi[-1])
        dirname = 'data_selection_' + str(start_frame) + '_to_' + str(end_frame)
        dirname = os.path.join(self.path, dirname)

        if os.path.exists(dirname):
            print 'Data selection path exists!'
        else:
            os.mkdir(dirname)

        image_sequence_dirname = 'image_sequence_' + str(start_frame) + '_to_' + str(end_frame)
        dirname = os.path.join(dirname, image_sequence_dirname)
        print 'Image sequence directory: ', dirname

        if os.path.exists(dirname):
            print 'Image selection path exists!'
        else:
            os.mkdir(dirname)

        print 'saving image sequence: ', len(self.image_sequence)
        zs = int(np.ceil( np.log10(len(self.image_sequence)) )+1)
        print 'zs: ', zs
        for i, image in enumerate(self.image_sequence):
            img_name = str(i).zfill(zs) + '.png'
            img_name = os.path.join(dirname, img_name)      
            cv2.imwrite(img_name, image)
            print i, img_name
        print 'To turn the PNGs into a movie, you can run this command from inside the directory with the tmp files: '
        print 'mencoder \'mf://*.png\' -mf type=png:fps=30 -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o animation.avi'
        print ' or '
        print "mencoder 'mf://*.jpg' -mf type=jpg:fps=30 -ovc x264 -x264encopts preset=slow:tune=film:crf=22 -oac copy -o animation.mp4"
        print "might need: https://www.faqforge.com/linux/how-to-install-ffmpeg-on-ubuntu-14-04/"
        print ''
        print '  or  '
        print "ffmpeg -n -i '"'animation_%04d.jpg'"' animation.m4v"
        print ' ^ that one is Mac compatible'
    def get_next_reconstructed_image(self):
        if len(self.image_sequence) < self.skip_frames:
            print('No images in sequence')
            return None, None
        self.current_frame += self.skip_frames
        if self.current_frame >= len(self.image_sequence)-1:
            self.current_frame = -1
        try:
            img = self.image_sequence[self.current_frame]      
            return self.image_sequence_timestamps[self.current_frame], img
        except:
            print('Missing frame? Frame: ' + str(self.current_frame) + ' skipping ahead.')
            return self.get_next_reconstructed_image()

    def updateData(self):
        if self.play:
            ## Display the data
            time_epoch, cvimg = self.get_next_reconstructed_image()
            if cvimg is None:
                print('Failed to play video.')
                self.play = False
                return
            try:
                self.img.setImage(cvimg)
            except AttributeError:
                self.init_bg_image()
                self.img.setImage(cvimg)
            
            QtCore.QTimer.singleShot(1, self.updateData)
            now = ptime.time()
            dt = (now-self.updateTime)
            self.updateTime = now
            
            if dt < self.frame_delay:
                d = self.frame_delay - dt
                time.sleep(d)
                
            self.current_time_vline.setPos(time_epoch)
            self.current_time_epoch = time_epoch
            
            del(cvimg)
    
    def run(self):
        ## Display the widget as a new window
        #self.w.show()
        ## Start the Qt event loop
        print 'Running!'
        self.show()
        
## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
        
    ## Read data #############################################################
    parser = OptionParser()
    parser.add_option('--path', type=str, default='none', help="option: path that points to standard named filename, background image, dvbag, config. If using 'path', no need to provide filename, bgimg, dvbag, and config. Note")
    parser.add_option('--movie', type=int, default=1, help="load and play the dvbag movie, default is 1, to load use 1")
    parser.add_option('--load-original', type=int, default=0, dest="load_original", help="load original (unprocessed) dataset for debugging, use 1 to load, default 0")
    parser.add_option('--draw-interesting-time-points', type=int, default=1, dest="draw_interesting_time_points", help="draw interesting time points (e.g. vertical lines). Default = True, set to False if VERY large dataset.")
    parser.add_option('--draw-config-function', type=int, default=0, dest="draw_config_function", help="If config has a draw function, apply this function to movie frames")
    parser.add_option('--clickable-width', type=int, default=6, dest="clickable_width", help="pixel distance from trace to accept click (larger number means easier to click traces)")
    parser.add_option('--filename', type=str, help="name and path of the hdf5 tracked_objects filename")
    parser.add_option('--bgimg', type=str, help="name and path of the background image")
    parser.add_option('--dvbag', type=str, default='none', help="name and path of the delta video bag file, optional")
    parser.add_option('--chunked_dvbag', type=int, default=0, help="if 1, look for a chunked bag created using chunk_bag.py, which works well for very large bags")
    parser.add_option('--config', type=str, default='none', help="name and path of a configuration file, optional. If the configuration file has an attribute 'sensory_stimulus_on', which should be a list of epoch timestamps e.g. [[t1,t2],[t3,4]], then these timeframes will be highlighted in the gui.")
    (options, args) = parser.parse_args()
    
    if options.path != 'none':
        if not os.path.isdir(options.path):
            raise ValueError('Path needs to be a directory!')
        options.filename = mta.read_hdf5_file_to_pandas.get_filename(options.path, 'trackedobjects.hdf5')
        options.config = mta.read_hdf5_file_to_pandas.get_filename(options.path, 'config')
        if options.dvbag == 'none':
            options.dvbag = mta.read_hdf5_file_to_pandas.get_filename(options.path, 'delta_video.bag')
        options.bgimg = mta.read_hdf5_file_to_pandas.get_filename(options.path, '_bgimg_')

    if options.movie != 1:
        options.dvbag = 'none'
    
    Qtrajec = QTrajectory(options.filename, options.bgimg, options.dvbag, options.load_original, options.clickable_width, 
        options.draw_interesting_time_points,
        options.draw_config_function,
        options.chunked_dvbag)
    Qtrajec.run()
    
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
        
'''

class MainWindow(TemplateBaseClass):  
    def __init__(self):
        TemplateBaseClass.__init__(self)
        self.setWindowTitle('pyqtgraph example: Qt Designer')
  
        # Create the main window
        self.ui = WindowTemplate()
        self.ui.setupUi(self)
        self.ui.movie_play.clicked.connect(self.plot)
  
        self.show()
  
    def plot(self):
        self.ui.qtplot_trajectory.plot(np.random.normal(size=100), clear=True)
  
  
  
## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    
    win = MainWindow()
    
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
'''
