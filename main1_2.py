#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 14:42:55 2026

@author: kensMACbook
"""


from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5 import *

from pyqtgraph import PlotWidget, plot, mkPen
import pyqtgraph as pg

import os, io, sys
import cv2
import numpy as np
import datetime, time
from core.Gpris import gpris
from makejson import BoxLabelingTool

# logger
from utils import utils
from utils.config import cfg_COMM
from utils.logger import get_log_view
from labelwindow import LabelWindow
from labeldialog import LabelDialog
from iirfilterdialog import IIRFilterDialog

from matplotlib import pyplot as plt

import json  # json 파일 읽기용
import platform

log_level = 1
log = get_log_view(log_level,  log_root=cfg_COMM.log_root, error_log=False, log_name='server_info') 
error_log = get_log_view(log_level, log_root=cfg_COMM.log_root, error_log=True, log_name='server_error')

form_class = uic.loadUiType("data/ui/main3_3.ui")[0]



class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
#        self.setWindowFlag(Qt.FramelessWindowHint)
        
        self.setupUi(self)
        
        
        ''' 상태변수 '''
        self.ch_value   = 0
        self.scan_value = 0
        self.data_value = 0
        
        ''' 데이터변수 '''
        self.t3r_data = None # (z, x, y)
        self.t3r_fname = None # t3r 파일명
        self.max_x = 0#24
        self.max_y = 0#1294
        self.max_z = 0#256
        self.heatmap_xy = None
        self.heatmap_zy = None
        self.heatmap_xz = None
        
        ''' 테스트 변수 '''
        self.is_equalize = False
        self.is_contrast = False
        self.is_clahe    = False
        
        self.is_procdisp = False
        """ 뷰 옵션 """
        self.scan_per_page = 300
        self.scan_start = 0
        self.scan_end = self.scan_per_page - 1
        self.sigrange = 5000
        
        '''라벨링용 변수'''
        self.ID = 0
        self.Class = "blink"
        self.start = [0.0,0.0,0.0]
        self.end = [0.1,0.1,0.1]
 
        """ UI 초기화 """
        self.init_ui_value()
        self.init_signal()
        
        """마우스 초기화"""
        self.m_flag=False
        
        # [1] 러버밴드(점선 사각형) 객체 생성
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.rubberBand1 = QRubberBand(QRubberBand.Rectangle, self)
        self.rubberBand2 = QRubberBand(QRubberBand.Rectangle, self)
        self.rubberBand3 = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = QPoint() # 드래그 시작점 저장용
        
        self.osYPoint = 20 if platform.system()=="Windows" else 0 # Menu 추가때문에 20pixel offset발생
        
 
    def init_ui_value(self):
        ''' 뷰 슬라이더 초기화 '''
        self.sl_ch.setRange(0, 1)
        self.sl_scan.setRange(0, 1)
        self.sl_data.setRange(0, 1)

        self.sl_drange.setRange(100, self.scan_per_page)
        self.sl_drange.setPageStep(10)
        self.sl_drange.setValue(self.scan_per_page)        
        self.txt_drange.setText(str(self.scan_per_page))

        self.sl_sigrange.setRange(100, 30000)
        self.sl_sigrange.setPageStep(100)
        self.sl_sigrange.setValue(self.sigrange)        
        self.txt_sigrange.setText(str(self.sigrange))

        ''' 뷰 슬라이더 text 초기화 '''
        self.txt_chval.setText(str(self.ch_value))
        self.txt_scanval.setText(str(self.scan_value))
        self.txt_dataval.setText(str(self.data_value))

        ''' 뷰 스크롤바 초기화 '''
#        self.scroll_scan.setMaximum(300)
#        self.scroll_scan.setPageStep(10)
        self.scroll_scan.hide()
        
        ''' AScanViwe 초기화 '''
#        self.AScanView.setConfigOptions(background = 'w')
        self.AScanView.showGrid(x=True, y=True)
        self.AScanView.setBackground('w')
        self.AScanView.getViewBox().invertY(True)
        self.AScanView.getViewBox().setBorder(mkPen(color = 'k', width=1))
        self.AScanView.showAxis('left', show=False)
        self.AScanView.showAxis('bottom', show=False)
#        self.AScanView.plot([30,32, 34, 32, 33, 31, 29, 32, 35, 45], [1,2,3,4,5,6,7,8,9,10])

        '''라벨링 초기화'''
        self.ID_lineEdit.setText(str(self.ID))
        self.Class_lineEdit.setText(str(self.Class))

        
    def init_signal(self):
        ''' t3r 데이터 추가버튼 '''
        self.btn_addfile.clicked.connect(self.select_data_file)
        ''' ch slider '''
        self.sl_ch.valueChanged.connect(self.change_sl_ch)
        ''' scan slider '''
        self.sl_scan.valueChanged.connect(self.change_sl_scan)
        ''' data(depth) slider '''
        self.txt_scanval.editingFinished.connect(self.change_txt_scan)
        self.sl_data.valueChanged.connect(self.change_sl_data)
        ''' drange(dispay) slider '''
        self.txt_chval.editingFinished.connect(self.change_txt_ch)
        self.txt_dataval.editingFinished.connect(self.change_txt_data)
        self.sl_drange.valueChanged.connect(self.change_sl_drange)
        ''' sigrange(dispay) slider '''
        self.sl_sigrange.valueChanged.connect(self.change_sl_sigrange)
        
        # ch idx -, + 버튼
        self.btn_chval_minus.clicked.connect(self.decrease_ch)
        self.btn_chval_plus.clicked.connect(self.increase_ch)

        # scan idx -, + 버튼
        self.btn_scan_minus.clicked.connect(self.decrease_scan)
        self.btn_scan_plus.clicked.connect(self.increase_scan)

        # data idx -, + 버튼
        self.btn_data_minus.clicked.connect(self.decrease_data)
        self.btn_data_plus.clicked.connect(self.increase_data)

        ''' 평단면 배치저장 '''
        self.btn_data_export.clicked.connect(self.save_batch_xy)
        ''' 종단면 배치저장 '''
        self.btn_ch_export.clicked.connect(self.save_batch_zy)
        ''' 횡단면 배치저장 '''
        self.btn_scan_export.clicked.connect(self.save_batch_xz)
        
        ''' 필터 테스트 '''
        self.test_equalize.stateChanged.connect(self.test_cb_equalize)
        self.test_contrast.stateChanged.connect(self.test_cb_contrast)
        self.test_clahe.stateChanged.connect(self.test_cb_clahe)

        self.chk_procdisp.stateChanged.connect(self.check_cb_procdisp)

        ''' 스크롤 테스트 '''
        self.scroll_scan.valueChanged.connect(self.change_scroll_scan)
        
        """ 메뉴 테스트 """
        self.actionOpen.triggered.connect(self.select_data_file)

        self.actionKalman_Filtering.triggered.connect(self.kalman_filtering)

        self.actionIIR_LPF.triggered.connect(self.IIR_LPF_filtering)
        self.actionIIR_HPF.triggered.connect(self.IIR_HPF_filtering)

        self.actionSobelXY.triggered.connect(self.SobelXY)
        self.actionSobelYZ.triggered.connect(self.SobelYZ)
        self.actionSobelXZ.triggered.connect(self.SobelXZ)

        self.actionGaussianXY.triggered.connect(self.GaussianXY)
        self.actionGaussianYZ.triggered.connect(self.GaussianYZ)
        self.actionGaussianXZ.triggered.connect(self.GaussianXZ)
        
        """라벨링"""
        self.btn_labeling.clicked.connect(self.save_label)
        self.ID_lineEdit.textChanged.connect(self.label_change) 
        self.Class_lineEdit.textChanged.connect(self.label_change) 
        self.btn_labelopen.clicked.connect(self.openlabel)

                # 하단 start/end 입력창 변경 시 self.start/self.end 업데이트
        self.startX_lineEdit.editingFinished.connect(self.update_start_from_ui)
        self.startY_lineEdit.editingFinished.connect(self.update_start_from_ui)
        self.startZ_lineEdit.editingFinished.connect(self.update_start_from_ui)
        self.endX_lineEdit.editingFinished.connect(self.update_end_from_ui)
        self.endY_lineEdit.editingFinished.connect(self.update_end_from_ui)
        self.endZ_lineEdit.editingFinished.connect(self.update_end_from_ui)

    # MOUSE Click drag EVENT function
    def mousePressEvent(self, event):
        if event.button()==Qt.LeftButton:
            realpos = event.pos() - QPoint(0,self.osYPoint) 
            #print(event.pos(), realpos)
            self.origin = event.pos()
                # 시작점과 크기가 0인 사각형으로 초기화 및 표시
            if QApplication.keyboardModifiers() == Qt.ControlModifier:
                self.rubberBand.setGeometry(QRect(self.origin, QSize()))
                self.rubberBand.setStyleSheet("selection-background-color: blue; border: 1px solid blue;")
                self.rubberBand.show()    
                if(self.lb_view_data.geometry().contains(realpos)):
                    self.set_ch_scan_from_position(realpos)    
                    self.start = [self.ch_value,self.scan_value,self.data_value]
                    self.show_startend()
                else:
                    if(self.lb_ch_data.geometry().contains(realpos)):
                        self.set_data_scan_from_position(realpos)
                        self.start = [self.ch_value,self.scan_value,self.data_value]
                        self.show_startend()
                    else:
                        if(self.lb_scan_data.geometry().contains(realpos)):
                            self.set_data_ch_from_position(realpos)
                            self.start = [self.ch_value,self.scan_value,self.data_value]
                            self.show_startend()
                        else:
                            self.m_flag=True
                            self.m_Position=event.globalPos()-self.pos() #Get the position of the mouse relative to the window
                            self.setCursor(QCursor(Qt.OpenHandCursor))  #Change mouse icon
                            event.accept()
                            self.start = [0.0,0.0,0.0]
                            self.end = [0.1,0.1,0.1]
                            self.show_startend()
                        
                            
            else:
                if(self.lb_view_data.geometry().contains(realpos)):
                    self.set_ch_scan_from_position(realpos)    
                    if self.end[0]==self.start[0]:
                        self.end[0] = self.ch_value
                        self.show_startend()
                        self.drawbox(self.start, self.end)
                    if self.end[1]==self.start[1]:
                        self.end[1] = self.scan_value
                        self.show_startend()
                        self.drawbox(self.start, self.end)
                else:
                    if(self.lb_ch_data.geometry().contains(realpos)):
                        self.set_data_scan_from_position(realpos)
                        if self.end[1]==self.start[1]:
                            self.end[1] = self.scan_value
                            self.show_startend()
                            self.drawbox(self.start, self.end)
                        if self.end[2]==self.start[2]:
                            self.end[2] = self.data_value
                            self.show_startend()
                            self.drawbox(self.start, self.end)
                    else:
                        if(self.lb_scan_data.geometry().contains(realpos)):
                            self.set_data_ch_from_position(realpos)
                            if self.end[0]==self.start[0]:
                                self.end[0] = self.ch_value
                                self.show_startend()
                                self.drawbox(self.start, self.end)
                            if self.end[2]==self.start[2]:
                                self.end[2] = self.data_value
                                self.show_startend()
                                self.drawbox(self.start, self.end)
                        else:
                            self.m_flag=True
                            self.m_Position=event.globalPos()-self.pos() #Get the position of the mouse relative to the window
                            self.setCursor(QCursor(Qt.OpenHandCursor))  #Change mouse icon
                            event.accept()
                            self.start = [0.0,0.0,0.0]
                            self.end = [0.1,0.1,0.1]
                            self.rubberBand1.hide()
                            self.rubberBand2.hide()
                            self.rubberBand3.hide()
                            self.show_startend()
                        
                        
    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:  
            self.move(QMouseEvent.globalPos()-self.m_Position)#Change window position
            self.setCursor(QCursor(Qt.OpenHandCursor))  #Change mouse icon
            QMouseEvent.accept()
        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            if self.rubberBand.isVisible():
                # 시작점(origin)과 현재 마우스 위치(event.pos())로 사각형 만듦
                # normalized()는 역방향(우->좌, 하->상) 드래그 시 좌표 오류 방지
                self.rubberBand.setGeometry(QRect(self.origin, QMouseEvent.pos()).normalized())
            
    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag=False
        self.setCursor(QCursor(Qt.ArrowCursor))
        #print("realease",QMouseEvent.pos())
        realpos = QMouseEvent.pos() - QPoint(0,self.osYPoint) 
        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            self.rubberBand.hide()
            self.drawbox(self.start, self.end)
            if(self.lb_view_data.geometry().contains(realpos)):
                x,y=self.return_ch_scan_from_position(realpos)
                self.end = [x,y,self.start[2]]
                self.show_startend()
                self.drawbox(self.start, self.end)
            else:
                if(self.lb_ch_data.geometry().contains(realpos)):
                    y,z=self.return_data_scan_from_position(realpos)
                    self.end = [self.start[0],y,z]
                    self.show_startend()
                    self.drawbox(self.start, self.end)
                else:
                    if(self.lb_scan_data.geometry().contains(realpos)):
                        z,x =self.return_data_ch_from_position(realpos)
                        self.end = [x,self.start[1],z]
                        self.show_startend()
                        self.drawbox(self.start, self.end)
                

    def wheelEvent(self, event) :
        if event.angleDelta().y() > 0 :
            self.scan_start += 10
            if self.scan_start > self.max_y - self.scan_per_page:
                self.scan_start =self.max_y - self.scan_per_page 
            
        if event.angleDelta().y() < 0 :
            self.scan_start -= 10
            if self.scan_start < 0:
                self.scan_start = 0

        self.scan_end = self.scan_start + self.scan_per_page - 1
        self.scroll_scan.setSliderPosition(self.scan_start)
        self.update_all_view()
        
    def set_ch_scan_from_position(self, realpos) :
        currentpos = realpos - self.lb_view_data.geometry().topLeft()
        ch = int(currentpos.y() *  self.max_x/ self.lb_view_data.geometry().height())
        scan = self.scan_start + int(currentpos.x() * self.scan_per_page / self.lb_view_data.geometry().width())
        self.change_ch_scan(ch, scan)

    def set_data_scan_from_position(self, realpos) :
        currentpos = realpos - self.lb_ch_data.geometry().topLeft()
        data = int(currentpos.y() *  self.max_z/ self.lb_ch_data.geometry().height())
        scan = self.scan_start + int(currentpos.x() * self.scan_per_page / self.lb_ch_data.geometry().width())
        self.change_data_scan(data, scan)

    def set_data_ch_from_position(self, realpos) :
        currentpos = realpos - self.lb_scan_data.geometry().topLeft()
        data = int(currentpos.y() *  self.max_z/ self.lb_scan_data.geometry().height())
        ch = int(currentpos.x() * self.max_x / self.lb_scan_data.geometry().width())
        self.change_data_ch(data, ch)
        
    def return_ch_scan_from_position(self, realpos) :
        currentpos = realpos - self.lb_view_data.geometry().topLeft()
        ch = int(currentpos.y() *  self.max_x/ self.lb_view_data.geometry().height())
        scan = self.scan_start + int(currentpos.x() * self.scan_per_page / self.lb_view_data.geometry().width())
        return(int(ch), int(scan))

    def return_data_scan_from_position(self, realpos) :
        currentpos = realpos - self.lb_ch_data.geometry().topLeft()
        data = int(currentpos.y() *  self.max_z/ self.lb_ch_data.geometry().height())
        scan = self.scan_start + int(currentpos.x() * self.scan_per_page / self.lb_ch_data.geometry().width())
        return(int(scan), int(data))

    def return_data_ch_from_position(self, realpos) :
        currentpos = realpos - self.lb_scan_data.geometry().topLeft()
        data = int(currentpos.y() *  self.max_z/ self.lb_scan_data.geometry().height())
        ch = int(currentpos.x() * self.max_x / self.lb_scan_data.geometry().width())
        return(int(data), int(ch))

    def test_cb_equalize(self):
        self.is_equalize = self.test_equalize.isChecked()
        self.init_all_view()
        
    def test_cb_contrast(self):
        self.is_contrast = self.test_contrast.isChecked()
        self.init_all_view()
        
    def test_cb_clahe(self):
        self.is_clahe = self.test_clahe.isChecked()
        self.init_all_view()

    def check_cb_procdisp(self):
        self.is_procdisp = self.chk_procdisp.isChecked()
        self.update_all_view()

    def select_data_file(self):
        log.info('select_data_file func')
#         self.selected_data_path = QFileDialog.getExistingDirectory(self, 'Select directory')
        self.selected_data_path = QFileDialog.getOpenFileName(self, 'Open file', './', 'GPR files (*.t3r *.t3d *.raw3d *.prc3d)')
        log.info('selected data path {} => {}'.format(type(self.selected_data_path), self.selected_data_path))
        try:
            self.t3r_fname = self.selected_data_path[0].split('/')[-1].split('.')[0]
        except Exception as e:
            error_log.error('test => {}'.format(e))
        
        if self.selected_data_path[0]:
            log.info('data select')
            
            self.t3r_data = gpris()
            print(self.selected_data_path[0])
            self.t3r_data.Open(self.selected_data_path[0])
            
            self.max_z, self.max_x, self.max_y = self.t3r_data.gprdata.shape
            
            self.set_scroll()
            self.set_slider()
                        # 같은 폴더/이름의 json 파일 읽기 (확장자만 .json)
            json_path = self.selected_data_path[0].replace('.prc3d', '.json')

            if os.path.exists(json_path):
                json_data = None
                encodings = ['utf-8-sig', 'cp949', 'euc-kr', 'latin1', 'iso-8859-1']

                for enc in encodings:
                    try:
                        with open(json_path, 'r', encoding=enc) as f:
                            json_data = json.load(f)
                        log.info(f"JSON 성공: 인코딩 = {enc}")
                        break
                    except UnicodeDecodeError:
                        log.info(f"인코딩 실패: {enc}")
                        continue
                    except json.JSONDecodeError as je:
                        log.error(f"JSON 파싱 에러 ({enc}): {je}")
                        break
                    except Exception as e:
                        log.error(f"기타 에러 ({enc}): {e}")
                        break

                if json_data is not None:
                    try:
                        target_pos = json_data.get('targetPosition', {})
                        target_type = json_data.get('targetType', 'unknown')

                        self.ch_value   = int(target_pos.get('x', 0))
                        self.scan_value = int(target_pos.get('y', 0))
                        self.data_value = int(target_pos.get('z', 0))

                        # UI 동기화
                        self.txt_chval.setText(str(self.ch_value))
                        self.txt_scanval.setText(str(self.scan_value))
                        self.txt_dataval.setText(str(self.data_value))
                        self.sl_ch.setValue(self.ch_value)
                        self.sl_scan.setValue(self.scan_value)
                        self.sl_data.setValue(self.data_value)

                        self.Class_lineEdit.setText(target_type)
                        self.Class_lineEdit.repaint()  # ← 강제 화면 갱신
                        QApplication.processEvents()   # ← 이벤트 처리 강제

                        # scan 범위 자동 조정
                        if self.scan_value < self.scan_start or self.scan_value > self.scan_end:
                            if self.scan_value < (self.max_y - self.scan_per_page):
                                self.scan_start = self.scan_value
                            else:
                                self.scan_start = max(0, self.max_y - self.scan_per_page)
                            self.scan_end = self.scan_start + self.scan_per_page - 1
                            if hasattr(self, 'scroll_scan') and self.scroll_scan.isVisible():
                                self.scroll_scan.setSliderPosition(self.scan_start)

                        log.info(f"JSON 적용 성공 → ch={self.ch_value}, scan={self.scan_value}, data={self.data_value}, type={target_type}")
                    except Exception as inner_e:
                        error_log.error(f"JSON 값 적용 중 에러: {inner_e}")
                else:
                    log.info("모든 인코딩 시도 실패 → 기본값(0) 사용")
            else:
                log.info("JSON 파일 없음 → 기본값 사용")

            self.setWindowTitle(f"MainWindow - {os.path.basename(self.selected_data_path[0])}")    
            self.init_all_view()
            
            log.info('gpris data load done')
            
        elif not self.selected_data_path[0]:
            log.info('none')

    def set_scroll(self):
        if (self.scan_per_page < self.max_y) :
            self.scroll_scan.show()
            self.scroll_scan.setPageStep(self.scan_per_page)
            self.scroll_scan.setMaximum(self.max_y - self.scan_per_page - 1)
            self.scroll_scan.setSliderPosition(self.scan_start)
        else:
            self.scroll_scan.hide()
        
    def set_slider(self):
        self.sl_ch.setRange(0, self.max_x-1)
        self.sl_scan.setRange(0, self.max_y-1)
        self.sl_data.setRange(0, self.max_z-1)
        self.sl_drange.setRange(100, self.max_y-1)
        self.sl_drange.setPageStep(100)

    
    def save_batch_xy(self):
        ''' 평단면 배치저장 '''
        try:
            if self.t3r_data is not None:
                min_idx = int(self.txt_data_start.text())
                max_idx = int(self.txt_data_end.text())
                run_idx = min_idx
                log.info('xy {} ~ {} batch save'.format(min_idx, max_idx))
                
                if min_idx <= max_idx:
                    save_root = os.path.join(cfg_COMM.batch_save_root, self.t3r_fname, cfg_COMM.xy_dir_name)
                    utils.make_folder(save_root)
                
                while run_idx <= max_idx and min_idx <= max_idx:
                    log.info('run_idx => {}'.format(run_idx))
                    
                    data_xy = self.t3r_data.gprdata[run_idx,:,:]
                    heatmap_xy = None
                    heatmap_xy = cv2.normalize(data_xy, heatmap_xy, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    heatmap_xy = cv2.applyColorMap(heatmap_xy, cv2.COLORMAP_BONE)
                    heatmap_xy = cv2.cvtColor(heatmap_xy, cv2.COLOR_BGR2GRAY)
                    
                    if self.is_contrast:
                        alpha = 1.0
                        heatmap_xy = np.clip((1+alpha)*heatmap_xy - 128*alpha, 0, 255).astype(np.uint8)
                    
                    if self.is_equalize:
                        heatmap_xy = cv2.equalizeHist(heatmap_xy)
                        
                    if self.is_clahe:
                        cl_val = float(self.test_cliplimit.text())
                        tile_val = int(self.test_tileGridSize.text())
                        clahe = cv2.createCLAHE(clipLimit=cl_val, tileGridSize=(tile_val, tile_val))
                        heatmap_xy = clahe.apply(heatmap_xy)
                    
                    fname = cfg_COMM.xy_img_format.format(self.t3r_fname, run_idx)
                    save_path = os.path.join(save_root, fname)
                    cv2.imwrite(save_path, heatmap_xy)
                    
                    run_idx += 1
        except Exception as e:
            error_log.error('error during save_batch_xy => {}'.format(e))
    
    def save_batch_zy(self):
        ''' 종단면 배치저장 '''
        try:
            if self.t3r_data is not None:
                min_idx = int(self.txt_ch_start.text())
                max_idx = int(self.txt_ch_end.text())
                run_idx = min_idx
                log.info('zy {} ~ {} batch save'.format(min_idx, max_idx))
                
                if min_idx <= max_idx:
                    save_root = os.path.join(cfg_COMM.batch_save_root, self.t3r_fname, cfg_COMM.zy_dir_name)
                    utils.make_folder(save_root)
                
                while run_idx <= max_idx and min_idx <= max_idx:
                    log.info('run_idx => {}'.format(run_idx))
                    
                    data_zy = self.t3r_data.gprdata[:,run_idx,:]
                    heatmap_zy = None
                    heatmap_zy = cv2.normalize(data_zy, heatmap_zy, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    heatmap_zy = cv2.applyColorMap(heatmap_zy, cv2.COLORMAP_BONE)
                    heatmap_zy = cv2.cvtColor(heatmap_zy, cv2.COLOR_BGR2GRAY)
                    
                    if self.is_contrast:
                        alpha = 1.0
                        heatmap_zy = np.clip((1+alpha)*heatmap_zy - 128*alpha, 0, 255).astype(np.uint8)
                    
                    if self.is_equalize:
                        heatmap_zy = cv2.equalizeHist(heatmap_zy)
                        
                    if self.is_clahe:
                        cl_val = float(self.test_cliplimit.text())
                        tile_val = int(self.test_tileGridSize.text())
                        clahe = cv2.createCLAHE(clipLimit=cl_val, tileGridSize=(tile_val, tile_val))
                        heatmap_zy = clahe.apply(heatmap_zy)
                    
                    fname = cfg_COMM.zy_img_format.format(self.t3r_fname, run_idx)
                    save_path = os.path.join(save_root, fname)
                    cv2.imwrite(save_path, heatmap_zy)
                    
                    run_idx += 1
        except Exception as e:
            error_log.error('error during save_batch_zy => {}'.format(e))
    
    def save_batch_xz(self):
        ''' 횡단면 배치저장 '''
        try:
            if self.t3r_data is not None:
                min_idx = int(self.txt_scan_start.text())
                max_idx = int(self.txt_scan_end.text())
                run_idx = min_idx
                log.info('xz {} ~ {} batch save'.format(min_idx, max_idx))
                
                if min_idx <= max_idx:
                    save_root = os.path.join(cfg_COMM.batch_save_root, self.t3r_fname, cfg_COMM.xz_dir_name)
                    utils.make_folder(save_root)
                
                while run_idx <= max_idx and min_idx <= max_idx:
                    log.info('run_idx => {}'.format(run_idx))
                    
                    
                    data_xz = self.t3r_data.gprdata[:,:,run_idx]
                    heatmap_xz = None
                    heatmap_xz = cv2.normalize(data_xz, heatmap_xz, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    heatmap_xz = cv2.applyColorMap(heatmap_xz, cv2.COLORMAP_BONE)
                    heatmap_xz = cv2.cvtColor(heatmap_xz, cv2.COLOR_BGR2GRAY)
                    
                    if self.is_contrast:
                        alpha = 1.0
                        heatmap_xz = np.clip((1+alpha)*heatmap_xz - 128*alpha, 0, 255).astype(np.uint8)
                    
                    if self.is_equalize:
                        heatmap_xz = cv2.equalizeHist(heatmap_xz)
                        
                    if self.is_clahe:
                        cl_val = float(self.test_cliplimit.text())
                        tile_val = int(self.test_tileGridSize.text())
                        clahe = cv2.createCLAHE(clipLimit=cl_val, tileGridSize=(tile_val, tile_val))
                        heatmap_xz = clahe.apply(heatmap_xz)
                    
                    fname = cfg_COMM.xz_img_format.format(self.t3r_fname, run_idx)
                    save_path = os.path.join(save_root, fname)
                    cv2.imwrite(save_path, heatmap_xz)
                    
                    run_idx += 1
        except Exception as e:
            error_log.error('error during save_batch_xz => {}'.format(e))
    
    ''' xy평면에서 Mouse click => 횡단면, 종단면 업데이트 '''
    def change_ch_scan(self, ch, scan):
        if self.t3r_data is not None:
            self.txt_chval.setText(str(ch))
            self.txt_scanval.setText(str(scan))
            self.sl_ch.setValue(ch)
            self.sl_scan.setValue(scan)

            self.ch_value = int(ch)
#            self.update_zy_heatmap()

            self.scan_value = int(scan)
#            self.update_xz_heatmap()
            
#            self.update_xy_heatmap()
            self.update_all_view()
   
    ''' yz평면에서 Mouse clike => 평단면, 종단면 업데이트 '''
    def change_data_scan(self, data, scan):
        if self.t3r_data is not None:
            self.txt_dataval.setText(str(data))
            self.txt_scanval.setText(str(scan))
            self.sl_data.setValue(data)
            self.sl_scan.setValue(scan)

            self.data_value = int(data)
#            self.update_xy_heatmap()

            self.scan_value = int(scan)
#            self.update_xz_heatmap()
            
#            self.update_zy_heatmap()
            self.update_all_view()
   
    ''' xz평면에서 Mouse click => 평단면, 횡단면 업데이트 '''
    def change_data_ch(self, data, ch):
        if self.t3r_data is not None:
            self.txt_chval.setText(str(ch))
            self.txt_dataval.setText(str(data))
            self.sl_ch.setValue(ch)
            self.sl_data.setValue(data)

            self.ch_value = int(ch)
#            self.update_zy_heatmap()

            self.data_value = int(data)
#            self.update_xy_heatmap()
            
#            self.update_xz_heatmap()
            self.update_all_view()
    
    ''' Scroll 업데이트 '''
    def change_scroll_scan(self):
        self.scan_start = self.scroll_scan.value()
        self.scan_end = self.scan_start + self.scan_per_page - 1
        self.update_all_view()
#        self.update_xy_heatmap()
#        self.update_zy_heatmap()
#        print(self.scroll_scan.value())
        
    ''' ch 값 => 종단면 업데이트 '''
    def change_sl_ch(self):
        self.txt_chval.setText(str(self.sl_ch.value()))
        
        if self.t3r_data is not None:
            self.ch_value = int(self.sl_ch.value())
#            self.update_zy_heatmap()
            self.update_all_view()
    
    ''' scan 값 => 횡단면 업데이트 '''
    def change_sl_scan(self):
        self.txt_scanval.setText(str(self.sl_scan.value()))
        
        if self.t3r_data is not None:
            self.scan_value = int(self.sl_scan.value())
#            self.update_xz_heatmap()
            if self.scan_value < self.scan_start or self.scan_value > self.scan_end:
                if self.scan_value < (self.max_y - self.scan_per_page) :
                    self.scan_start = self.scan_value
                else:
                    self.scan_start = self.max_y - self.scan_per_page -1
                self.scan_end = self.scan_start + self.scan_per_page - 1
#                self.update_xy_heatmap()
#                self.update_zy_heatmap()
            self.update_all_view()

    def change_txt_scan(self):
        try:
            new_scan = int(self.txt_scanval.text().strip())
            if 0 <= new_scan < self.max_y:
                self.scan_value = new_scan
                self.sl_scan.setValue(new_scan)

                # 표시 범위 조정 (스크롤이 필요한 경우)
                if new_scan < self.scan_start or new_scan > self.scan_end:
                    if new_scan < (self.max_y - self.scan_per_page):
                        self.scan_start = new_scan
                    else:
                        self.scan_start = self.max_y - self.scan_per_page - 1
                    self.scan_end = self.scan_start + self.scan_per_page - 1
                    self.scroll_scan.setSliderPosition(self.scan_start)

                self.update_all_view()
            else:
                self.txt_scanval.setText(str(self.scan_value))
        except ValueError:
            self.txt_scanval.setText(str(self.scan_value))
                

    def change_txt_ch(self):
        try:
            new_ch = int(self.txt_chval.text().strip())
            if 0 <= new_ch < self.max_x:
                self.ch_value = new_ch
                self.sl_ch.setValue(new_ch)
                self.update_all_view()
            else:
                self.txt_chval.setText(str(self.ch_value))
                print(f"ch 입력 범위 초과: {new_ch} (max_x={self.max_x})")
        except ValueError:
            self.txt_chval.setText(str(self.ch_value))
            print("ch 입력값이 숫자가 아님")

    def change_txt_data(self):
        try:
            new_data = int(self.txt_dataval.text().strip())
            if 0 <= new_data < self.max_z:
                self.data_value = new_data
                self.sl_data.setValue(new_data)
                self.update_all_view()
            else:
                self.txt_dataval.setText(str(self.data_value))
                print(f"data 입력 범위 초과: {new_data} (max_z={self.max_z})")
        except ValueError:
            self.txt_dataval.setText(str(self.data_value))
            print("data 입력값이 숫자가 아님")


    ''' data 값 => 평단면 업데이트 '''
    def change_sl_data(self):
        self.txt_dataval.setText(str(self.sl_data.value()))
        
        if self.t3r_data is not None:
            self.data_value = int(self.sl_data.value())
#            self.update_xy_heatmap()
            self.update_all_view()

    def change_sl_drange(self):
        self.txt_drange.setText(str(self.sl_drange.value()))
        
        if self.t3r_data is not None:
            self.scan_per_page = int(self.sl_drange.value())
            self.scan_end = self.scan_start + self.scan_per_page - 1
            if self.scan_end > self.max_y - 1:
                self.scan_end = self.max_y -1
                self.scan_start = self.scan_end - self.scan_per_page
                print(self.scan_start, self.scan_end, self.scan_per_page)           
            self.set_scroll()
            self.update_all_view()

    def change_sl_sigrange(self):
        self.txt_sigrange.setText(str(self.sl_sigrange.value()))
        
        if self.t3r_data is not None:
            self.sigrange = int(self.sl_sigrange.value())
#            self.update_xy_heatmap()
            self.update_all_view()
        
    def update_xy_heatmap(self):
        ''' 평단면 업데이트 '''
        try:
            if self.is_procdisp:
                data_xy = self.t3r_data.procdata[self.data_value,:,self.scan_start:self.scan_end]
            else :
                data_xy = self.t3r_data.gprdata[self.data_value,:,self.scan_start:self.scan_end]
            s_data_xy = data_xy * 128.0/self.sigrange + 128
            ss_data_xy = np.clip(s_data_xy, 0, 255)
            self.heatmap_xy = ss_data_xy.astype(np.uint8)
#            self.heatmap_xy = None
#            self.heatmap_xy = cv2.normalize(ss_data_xy, self.heatmap_xy, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            self.heatmap_xy = cv2.applyColorMap(self.heatmap_xy, cv2.COLORMAP_BONE)
            self.heatmap_xy = cv2.cvtColor(self.heatmap_xy, cv2.COLOR_BGR2GRAY)
#            self.heatmap_xy = cv2.line(self.heatmap_xy, (self.scan_value, 0), (self.scan_value, self.max_x), (255,255,255))
#            self.heatmap_xy = cv2.line(self.heatmap_xy, (0, self.ch_value), (self.max_y, self.ch_value), (255,255,255))
            if self.is_contrast:
                alpha = 1.0
                self.heatmap_xy = np.clip((1+alpha)*self.heatmap_xy - 128*alpha, 0, 255).astype(np.uint8)
            
            if self.is_equalize:
                self.heatmap_xy = cv2.equalizeHist(self.heatmap_xy)
                
            if self.is_clahe:
                cl_val = float(self.test_cliplimit.text())
                tile_val = int(self.test_tileGridSize.text())
                clahe = cv2.createCLAHE(clipLimit=cl_val, tileGridSize=(tile_val, tile_val))
                self.heatmap_xy = clahe.apply(self.heatmap_xy)
            
        except Exception as e:
            error_log.error('error during xy update cvt color => {}'.format(e))
        
        ''' #1 heatmap xy '''
        try:
            view_w, view_h = self.lb_view_data.geometry().width(), self.lb_view_data.geometry().height()
            resized_heatmap_xy = self.heatmap_xy.copy()
            resized_heatmap_xy = cv2.resize(resized_heatmap_xy, dsize=(view_w, view_h), interpolation=cv2.INTER_NEAREST)
            img_h, img_w = resized_heatmap_xy.shape
            
            bytesPerLine = 1 * img_w # 2 x, 
            qImg = QImage(resized_heatmap_xy.data, img_w, img_h, bytesPerLine, QImage.Format_Grayscale8)# .rgbSwapped() Grayscale8  Format_RGB888
            self.xyQP = QPixmap(qImg)
            self.draw_hairline_xyplane()
            self.lb_view_data.setPixmap(self.xyQP)
        except Exception as e:
            error_log.error('error during xy update apply pixmap => {}'.format(e))
    
    def draw_hairline_xyplane(self):
        qp = QPainter(self.xyQP)
        qp.setPen(QPen(Qt.yellow, 1))
        hairline_x = int((self.scan_value - self.scan_start+ 0.5) * self.lb_view_data.geometry().width() / (self.scan_per_page))
        hairline_y = int((self.ch_value+ 0.5) * self.lb_view_data.geometry().height() / (self.max_x))
        qp.drawLine(hairline_x,0, hairline_x, self.lb_view_data.geometry().height())
        qp.drawLine(0, hairline_y, self.lb_view_data.geometry().width(), hairline_y)
        qp.end()
        
    def update_zy_heatmap(self):
        ''' 종단면 업데이트 '''
        try:
            if self.is_procdisp:
                data_zy = self.t3r_data.procdata[:,self.ch_value,self.scan_start:self.scan_end]
            else :
                data_zy = self.t3r_data.gprdata[:,self.ch_value,self.scan_start:self.scan_end]
            s_data_zy = data_zy * 128.0/self.sigrange + 128
            ss_data_zy = np.clip(s_data_zy, 0, 255)
            self.heatmap_zy = ss_data_zy.astype(np.uint8)
#            self.heatmap_zy = None
#            self.heatmap_zy = cv2.normalize(data_zy, self.heatmap_zy, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            self.heatmap_zy = cv2.applyColorMap(self.heatmap_zy, cv2.COLORMAP_BONE)
            self.heatmap_zy = cv2.cvtColor(self.heatmap_zy, cv2.COLOR_BGR2GRAY)
            
            if self.is_contrast:
                alpha = 1.0
                self.heatmap_zy = np.clip((1+alpha)*self.heatmap_zy - 128*alpha, 0, 255).astype(np.uint8)
            
            if self.is_equalize:
                self.heatmap_zy = cv2.equalizeHist(self.heatmap_zy)
                
            if self.is_clahe:
                cl_val = float(self.test_cliplimit.text())
                tile_val = int(self.test_tileGridSize.text())
                clahe = cv2.createCLAHE(clipLimit=cl_val, tileGridSize=(tile_val, tile_val))
                self.heatmap_zy = clahe.apply(self.heatmap_zy)
                
        except Exception as e:
            error_log.error('error during zy update cvt color => {}'.format(e))
        
        ''' #2 heatmap zy '''
        try:
            view_w, view_h = self.lb_ch_data.geometry().width(), self.lb_ch_data.geometry().height()
            resized_heatmap_zy = self.heatmap_zy.copy()
            resized_heatmap_zy = cv2.resize(resized_heatmap_zy, dsize=(view_w, view_h), interpolation=cv2.INTER_NEAREST)
            img_h, img_w = resized_heatmap_zy.shape
            
            bytesPerLine = 1 * img_w # 2 x, 
            qImg = QImage(resized_heatmap_zy.data, img_w, img_h, bytesPerLine, QImage.Format_Grayscale8)# .rgbSwapped() Grayscale8  Format_RGB888
            self.zyQP = QPixmap(qImg)
            self.draw_hairline_zyplane()
            self.lb_ch_data.setPixmap(self.zyQP)
        except Exception as e:
            error_log.error('error during zy update apply pixmap => {}'.format(e))

    def draw_hairline_zyplane(self):
        qp = QPainter(self.zyQP)
        qp.setPen(QPen(Qt.yellow, 1))
        hairline_x = int((self.scan_value - self.scan_start + 0.5) * self.lb_ch_data.geometry().width() / (self.scan_per_page))
        hairline_y = int((self.data_value+ 0.5) * self.lb_ch_data.geometry().height() / (self.max_z))
        qp.drawLine(hairline_x,0, hairline_x, self.lb_ch_data.geometry().height())
        qp.drawLine(0, hairline_y, self.lb_ch_data.geometry().width(), hairline_y)
        qp.end()
        
    
    def update_xz_heatmap(self):
        ''' 횡단면 업데이트 '''
        try:
            if self.is_procdisp:
                data_xz = self.t3r_data.procdata[:,:,self.scan_value]
            else :
                data_xz = self.t3r_data.gprdata[:,:,self.scan_value]
            s_data_xz = data_xz * 128.0/self.sigrange + 128
            ss_data_xz = np.clip(s_data_xz, 0, 255)
            self.heatmap_xz = ss_data_xz.astype(np.uint8)
#            self.heatmap_xz = None
#            self.heatmap_xz = cv2.normalize(data_xz, self.heatmap_xz, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            self.heatmap_xz = cv2.applyColorMap(self.heatmap_xz, cv2.COLORMAP_BONE)
            self.heatmap_xz = cv2.cvtColor(self.heatmap_xz, cv2.COLOR_BGR2GRAY)
            
            if self.is_contrast:
                alpha = 1.0
                self.heatmap_xz = np.clip((1+alpha)*self.heatmap_xz - 128*alpha, 0, 255).astype(np.uint8)
            
            if self.is_equalize:
                self.heatmap_xz = cv2.equalizeHist(self.heatmap_xz)
                
            if self.is_clahe:
                cl_val = float(self.test_cliplimit.text())
                tile_val = int(self.test_tileGridSize.text())
                clahe = cv2.createCLAHE(clipLimit=cl_val, tileGridSize=(tile_val, tile_val))
                self.heatmap_xz = clahe.apply(self.heatmap_xz)
                
        except Exception as e:
            error_log.error('error during xz update cvt color => {}'.format(e))
        
        ''' #2 heatmap xz '''
        try:
            view_w, view_h = self.lb_scan_data.geometry().width(), self.lb_scan_data.geometry().height()
            resized_heatmap_xz = self.heatmap_xz.copy()
            resized_heatmap_xz = cv2.resize(resized_heatmap_xz, dsize=(view_w, view_h), interpolation=cv2.INTER_NEAREST)
            img_h, img_w = resized_heatmap_xz.shape
            
            bytesPerLine = 1 * img_w # 2 x, 
            qImg = QImage(resized_heatmap_xz.data, img_w, img_h, bytesPerLine, QImage.Format_Grayscale8)# .rgbSwapped() Grayscale8  Format_RGB888
            self.xzQP = QPixmap(qImg)
            self.draw_hairline_xzplane()
            self.lb_scan_data.setPixmap(self.xzQP)
        except Exception as e:
            error_log.error('error during xz update apply pixmap => {}'.format(e))

    def draw_hairline_xzplane(self):
        qp = QPainter(self.xzQP)
        qp.setPen(QPen(Qt.yellow, 1))
        hairline_x = int((self.ch_value + 0.5)* self.lb_scan_data.geometry().width() / (self.max_x))
        hairline_y = int((self.data_value + 0.5)* self.lb_scan_data.geometry().height() / (self.max_z))
        qp.drawLine(hairline_x,0, hairline_x, self.lb_scan_data.geometry().height())
        qp.drawLine(0, hairline_y, self.lb_scan_data.geometry().width(), hairline_y)
        qp.end()
        
    def update_all_view(self):
        if self.t3r_data is not None:
            self.update_xy_heatmap()
            ''' 종단면 초기화 '''
            self.update_zy_heatmap()
            ''' 횡단면 초기화 '''
            self.update_xz_heatmap()
            self.AscanPlot()      
        
    def init_all_view(self):
        log.info('================> z, x, y shape => {}'.format(self.t3r_data.gprdata.shape))
        ''' 평단면 초기화 '''
#        self.update_xy_heatmap()
        ''' 종단면 초기화 '''
#        self.update_zy_heatmap()
        ''' 횡단면 초기화 '''
#        self.update_xz_heatmap()
        self.update_all_view()

        
#         cv2.imwrite("heatmap_xy.png", self.heatmap_xy)
#         cv2.imwrite("heatmap_zy.png", self.heatmap_zy)
#         cv2.imwrite("heatmap_xz.png", self.heatmap_xz)
        log.info('init_all_view done')
        
    def contextMenuEvent(self, event):
        realpos = event.pos() - QPoint(0, 20) # Menu 추가때문에 25pixel offset발생
#            print(event.pos(), realpos)
        if(self.lb_view_data.geometry().contains(realpos)):
            self.set_ch_scan_from_position(realpos)
            contextMenu = QMenu(self)
            openAct = contextMenu.addAction("Open")
            poiXYAddAct = contextMenu.addAction("POI XY Add")
            poiInfoAct = contextMenu.addAction("POI Info")
            action = contextMenu.exec_(self.mapToGlobal(event.pos()))
            if action == openAct:
                self.select_data_file()
            if action == poiXYAddAct:
                self.poiXYAdd()
            
        if(self.lb_ch_data.geometry().contains(realpos)):
            self.set_data_scan_from_position(realpos)
            contextMenu = QMenu(self)
            openAct = contextMenu.addAction("Open")
            poiYZAddAct = contextMenu.addAction("POI YZ Add")
            poiInfoAct = contextMenu.addAction("POI Info")
            action = contextMenu.exec_(self.mapToGlobal(event.pos()))
            if action == openAct:
                self.select_data_file()
            if action == poiYZAddAct:
                self.poiYZAdd()
            
        if(self.lb_scan_data.geometry().contains(realpos)):
            self.set_data_ch_from_position(realpos)
            contextMenu = QMenu(self)
            openAct = contextMenu.addAction("Open")
            poiXZAddAct = contextMenu.addAction("POI XZ Add")
            poiInfoAct = contextMenu.addAction("POI Info")
            action = contextMenu.exec_(self.mapToGlobal(event.pos()))
            if action == openAct:
                self.select_data_file()
            if action == poiXZAddAct:
                self.poiXZAdd()

    def poiXYAdd(self):
        print("POI XY Add Selected")
#static method를 통해 dialog실행방법
#        result = LabelWindow.setLabelWindow()
        
#        if result :
#            print("Ok button is clicked")
#        else:
#            print("Cancel button is clicked")

#직접 dialog실행
#        dlg = LabelWindow(self.ch_value, self.scan_value, self.data_value) #부모상속 없이 다이얼로그 띄울때
        dlg = LabelDialog(self, 0) #부모 상속포함시
#        result = dlg.exec_()
        result = dlg.showModal()
        if result:
            label_name = dlg.label_name
            print(label_name)
        else:
            print("cancel button is clicked")

    def poiYZAdd(self):
        print("POI YZ Add Selected")
        dlg = LabelDialog(self, 1) #부모 상속포함시
#        result = dlg.exec_()
        result = dlg.showModal()
        if result:
            label_name = dlg.label_name
            print(label_name)
        else:
            print("cancel button is clicked")

    def poiXZAdd(self):
        print("POI XZ Add Selected")
        dlg = LabelDialog(self, 2) #부모 상속포함시
#        result = dlg.exec_()
        result = dlg.showModal()
        if result:
            label_name = dlg.label_name
            print(label_name)
        else:
            print("cancel button is clicked")

    
    def kalman_filtering(self) :
#        self.t3r_data.Kalman1DFilter(0.1, 1e-7, 0)
        self.t3r_data.Kalman2DFilter(0.1, 1e-7, 0)
        self.is_procdisp = True;
        if self.chk_procdisp.isChecked():
            print("Checkbox is already checked")
        else :
            self.chk_procdisp.toggle()
        self.update_all_view()

    def IIR_LPF_filtering(self) :
        print("IIR_LPF filter is applied")
        dlg = IIRFilterDialog(0, 1.0, 2) #부모 상속포함시
        result = dlg.showModal()
        if result:
            if self.chk_procdisp.isChecked():
                self.t3r_data.IIR_ButterLPF(dlg.cutoffFreq, dlg.filterOrder, 1)  #1GHz LPF
            else :
                self.t3r_data.IIR_ButterLPF(dlg.cutoffFreq, dlg.filterOrder, 0)  #1GHz LPF
    
            if self.chk_procdisp.isChecked():
                print("Checkbox is already checked")
            else :
                self.chk_procdisp.toggle()
            self.update_all_view()
        else:
            print("Cancel button is selected")
            
    def IIR_HPF_filtering(self) :
        print("IIR_HPF filter is applied")
        dlg = IIRFilterDialog(1, 0.3, 2) #부모 상속포함시
        result = dlg.showModal()
        if result:
            print("OK button is selected")
            if self.chk_procdisp.isChecked():
                self.t3r_data.IIR_ButterHPF(dlg.cutoffFreq, dlg.filterOrder, 1)  #1GHz LPF
            else :
                self.t3r_data.IIR_ButterHPF(dlg.cutoffFreq, dlg.filterOrder, 0)  #1GHz LPF
    
            if self.chk_procdisp.isChecked():
                print("Checkbox is already checked")
            else :
                self.chk_procdisp.toggle()
            self.update_all_view()
        else:
            print("Cancel button is selected")
        
        
    def SobelXY(self) :
        print("Sobel XY filter is applied")
        if self.chk_procdisp.isChecked():
            self.t3r_data.Sobel_XY(3, 1)
        else :
            self.t3r_data.Sobel_XY(3, 0)

        if self.chk_procdisp.isChecked():
            print("Checkbox is already checked")
        else :
            self.chk_procdisp.toggle()
        self.update_all_view()
        
    def SobelYZ(self) :
        print("Sobel YZ filter is applied")
        if self.chk_procdisp.isChecked():
            self.t3r_data.Sobel_YZ(3, 1)
        else :
            self.t3r_data.Sobel_YZ(3, 0)

        if self.chk_procdisp.isChecked():
            print("Checkbox is already checked")
        else :
            self.chk_procdisp.toggle()
        self.update_all_view()
        
    def SobelXZ(self) :
        print("Sobel XZ filter is applied")
        if self.chk_procdisp.isChecked():
            self.t3r_data.Sobel_XZ(3, 1)
        else :
            self.t3r_data.Sobel_XZ(3, 0)

        if self.chk_procdisp.isChecked():
            print("Checkbox is already checked")
        else :
            self.chk_procdisp.toggle()
        self.update_all_view()
        
    def GaussianXY(self) :
        print("Gaussian XY filter is applied")
        if self.chk_procdisp.isChecked():
            self.t3r_data.Gaussian_XY(3, 1)
        else :
            self.t3r_data.Gaussian_XY(3, 0)

        if self.chk_procdisp.isChecked():
            print("Checkbox is already checked")
        else :
            self.chk_procdisp.toggle()
        self.update_all_view()
        
        
    def GaussianYZ(self) :
        print("Gaussian YZ filter is applied")
        if self.chk_procdisp.isChecked():
            self.t3r_data.Gaussian_YZ(3, 1)
        else :
            self.t3r_data.Gaussian_YZ(3, 0)

        if self.chk_procdisp.isChecked():
            print("Checkbox is already checked")
        else :
            self.chk_procdisp.toggle()
        self.update_all_view()
        
        
    def GaussianXZ(self) :
        print("Gaussian XZ filter is applied")
        if self.chk_procdisp.isChecked():
            self.t3r_data.Gaussian_XZ(3, 1)
        else :
            self.t3r_data.Gaussian_XZ(3, 0)

        if self.chk_procdisp.isChecked():
            print("Checkbox is already checked")
        else :
            self.chk_procdisp.toggle()
        self.update_all_view()
        
        
    def AscanPlot(self) :
        if self.t3r_data is not None:
            ydata = np.array(range(0, 256))
            if self.chk_procdisp.isChecked():
                xdata = self.t3r_data.procdata[:, self.ch_value, self.scan_value]
            else :
                xdata = self.t3r_data.gprdata[:, self.ch_value, self.scan_value]
            self.AScanView.clear()
            self.AScanView.plot(xdata, ydata, pen = pg.mkPen(color = 'r', width = 2.0))
            self.AScanView.getViewBox().setXRange(-self.sigrange, self.sigrange, padding=0)
            self.AScanView.getViewBox().setYRange(ydata[0], ydata[255], padding=0)
            self.AScanView.addLine(y=self.data_value, pen = pg.mkPen(color = 'g', width = 1.0))
            
        
    def label_change(self):
        self.ID=self.ID_lineEdit.text() 
        self.Class=self.Class_lineEdit.text() 
        #print("ID:{},Class:{}".format(self.ID,self.Class))
        
    def show_startend(self):
        self.startX_lineEdit.setText(str(self.start[0]))
        self.startY_lineEdit.setText(str(self.start[1]))
        self.startZ_lineEdit.setText(str(self.start[2]))
        
        self.endX_lineEdit.setText(str(self.end[0]))
        self.endY_lineEdit.setText(str(self.end[1]))
        self.endZ_lineEdit.setText(str(self.end[2]))
        
    def drawsqure1(self, start,end):
        self.rubberBand1.setGeometry(QRect(QPoint(start[0],start[1])+QPoint(0,self.osYPoint), QSize()))
        self.rubberBand1.setGeometry(QRect(QPoint(start[0],start[1])+QPoint(0,self.osYPoint), 
                                    QPoint(end[0],end[1])+QPoint(0,self.osYPoint)).normalized())
        self.rubberBand1.setStyleSheet("selection-background-color: red; border: 1px solid red;")
        self.rubberBand1.show()  
        
    def drawsqure2(self, start,end):
        self.rubberBand2.setGeometry(QRect(QPoint(start[0],start[1])+QPoint(0,self.osYPoint), QSize()))
        self.rubberBand2.setGeometry(QRect(QPoint(start[0],start[1])+QPoint(0,self.osYPoint), 
                                    QPoint(end[0],end[1])+QPoint(0,self.osYPoint)).normalized())
        self.rubberBand2.setStyleSheet("selection-background-color: red; border: 1px solid red;")
        self.rubberBand2.show()  
        
    def drawsqure3(self, start,end):
        self.rubberBand3.setGeometry(QRect(QPoint(start[0],start[1])+QPoint(0,self.osYPoint), QSize()))
        self.rubberBand3.setGeometry(QRect(QPoint(start[0],start[1])+QPoint(0,self.osYPoint), 
                                    QPoint(end[0],end[1])+QPoint(0,self.osYPoint)).normalized())
        self.rubberBand3.setStyleSheet("selection-background-color: red; border: 1px solid red;")
        self.rubberBand3.show()  
        
    def drawbox(self, start, end):
        ys1 = start[0]*self.lb_view_data.geometry().height()/self.max_x + self.lb_view_data.geometry().topLeft().y()
        xs1 = start[1]*self.lb_view_data.geometry().width()/self.scan_per_page + self.lb_view_data.geometry().topLeft().x()
        ye1 = end[0]*self.lb_view_data.geometry().height()/self.max_x + self.lb_view_data.geometry().topLeft().y()
        xe1 = end[1]*self.lb_view_data.geometry().width()/self.scan_per_page + self.lb_view_data.geometry().topLeft().x()
        self.drawsqure1([int(xs1),int(ys1)], [int(xe1),int(ye1)])
        ys2 = start[2]*self.lb_ch_data.geometry().height()/self.max_z + self.lb_ch_data.geometry().topLeft().y()
        xs2 = start[1]*self.lb_ch_data.geometry().width()/self.scan_per_page + self.lb_ch_data.geometry().topLeft().x()
        ye2 = end[2]*self.lb_ch_data.geometry().height()/self.max_z + self.lb_ch_data.geometry().topLeft().y()
        xe2 = end[1]*self.lb_ch_data.geometry().width()/self.scan_per_page + self.lb_ch_data.geometry().topLeft().x()
        self.drawsqure2([int(xs2),int(ys2)], [int(xe2),int(ye2)])
        ys3 = start[2]*self.lb_scan_data.geometry().height()/self.max_z + self.lb_scan_data.geometry().topLeft().y()
        xs3 = start[0]*self.lb_scan_data.geometry().width()/self.max_x + self.lb_scan_data.geometry().topLeft().x()
        ye3 = end[2]*self.lb_scan_data.geometry().height()/self.max_z + self.lb_scan_data.geometry().topLeft().y()
        xe3 = end[0]*self.lb_scan_data.geometry().width()/self.max_x + self.lb_scan_data.geometry().topLeft().x()
        self.drawsqure3([int(xs3),int(ys3)], [int(xe3),int(ye3)])
        
    def openlabel(self):
        filename = QFileDialog.getOpenFileName(self, 'Open file', './', ' (*.json)')
        label = BoxLabelingTool()
        ID,Class,start,end=label.readjson(filename[0])
        self.ID_lineEdit.setText(str(ID))
        self.Class_lineEdit.setText(str(Class))
        self.start = start
        self.ch_value = start[0]
        self.scan_value = start[1]
        self.data_value = start[2]
        self.txt_scanval.setText(str(self.scan_value))
        self.txt_chval.setText(str(self.ch_value))
        self.txt_dataval.setText(str(self.data_value))
        self.end = end
        self.update_all_view()
        self.show_startend()
        self.drawbox(self.start, self.end)
        
    def decrease_ch(self):
        if self.ch_value > 0:
            self.ch_value -= 1
            self.sl_ch.setValue(self.ch_value)
            self.txt_chval.setText(str(self.ch_value))
            self.update_all_view()

    def increase_ch(self):
        if self.ch_value < self.max_x - 1:
            self.ch_value += 1
            self.sl_ch.setValue(self.ch_value)
            self.txt_chval.setText(str(self.ch_value))
            self.update_all_view()

    def decrease_scan(self):
        if self.scan_value > 0:
            self.scan_value -= 1
            self.sl_scan.setValue(self.scan_value)
            self.txt_scanval.setText(str(self.scan_value))
            # 범위 조정
            if self.scan_value < self.scan_start:
                self.scan_start = max(0, self.scan_value)
                self.scan_end = self.scan_start + self.scan_per_page - 1
                self.scroll_scan.setSliderPosition(self.scan_start)
            self.update_all_view()

    def increase_scan(self):
        if self.scan_value < self.max_y - 1:
            self.scan_value += 1
            self.sl_scan.setValue(self.scan_value)
            self.txt_scanval.setText(str(self.scan_value))
            if self.scan_value > self.scan_end:
                self.scan_start = min(self.max_y - self.scan_per_page, self.scan_value - self.scan_per_page + 1)
                self.scan_end = self.scan_start + self.scan_per_page - 1
                self.scroll_scan.setSliderPosition(self.scan_start)
            self.update_all_view()

    def decrease_data(self):
        if self.data_value > 0:
            self.data_value -= 1
            self.sl_data.setValue(self.data_value)
            self.txt_dataval.setText(str(self.data_value))
            self.update_all_view()

    def increase_data(self):
        if self.data_value < self.max_z - 1:
            self.data_value += 1
            self.sl_data.setValue(self.data_value)
            self.txt_dataval.setText(str(self.data_value))
            self.update_all_view()   
        
    def update_start_from_ui(self):
        try:
            self.start[0] = int(self.startX_lineEdit.text())
            self.start[1] = int(self.startY_lineEdit.text())
            self.start[2] = int(self.startZ_lineEdit.text())
            print("start 업데이트 (정수):", self.start)  # 디버깅용
        except ValueError:
            print("start 입력값 오류 - 정수만 입력하세요")
            # 필요하면 원래 값으로 되돌리기
            self.show_startend()  # 입력창을 원래 값으로 복구

    def update_end_from_ui(self):
        try:
            self.end[0] = int(self.endX_lineEdit.text())
            self.end[1] = int(self.endY_lineEdit.text())
            self.end[2] = int(self.endZ_lineEdit.text())
            print("end 업데이트 (정수):", self.end)  # 디버깅용
        except ValueError:
            print("end 입력값 오류 - 정수만 입력하세요")
            self.show_startend()  # 원래 값 복구

    def save_label(self):
        # UI 최신 값 반영
        self.update_start_from_ui()
        self.update_end_from_ui()

        # ID, Class 가져오기
        self.ID = int(self.ID_lineEdit.text() or 0)
        self.Class = self.Class_lineEdit.text().strip() or "unknown"

        # start/end → center + dimensions 변환 (정수 유지)
        start = [int(v) for v in self.start]
        end   = [int(v) for v in self.end]

        center = [
            (start[0] + end[0]) / 2.0,
            (start[1] + end[1]) / 2.0,
            (start[2] + end[2]) / 2.0
        ]

        dimensions = [
            abs(end[0] - start[0]),  # length (x 방향)
            abs(end[1] - start[1]),  # width  (y 방향)
            abs(end[2] - start[2])   # height (z 방향)
        ]

        # json 데이터 (원래 양식 맞춤)
        data = {
            "id": self.ID,
            "class": self.Class,          # 소문자 class (원본 예시 따라)
            "center": {
                "x": center[0],
                "y": center[1],
                "z": center[2]
            },
            "dimensions": {
                "length": dimensions[0],
                "width":  dimensions[1],
                "height": dimensions[2]
            }
        }

        # 저장 경로: 원본 파일과 같은 폴더 + _bbox 붙임
        if hasattr(self, 'selected_data_path') and self.selected_data_path and self.selected_data_path[0]:
            base = os.path.splitext(self.selected_data_path[0])[0]
            json_path = base + "_bbox.json"
        else:
            json_path = os.path.join(os.getcwd(), f"label_{self.ID}_{self.Class}_bbox.json")
            print("경고: 원본 경로 없음 → 현재 폴더 저장")

        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"저장 완료: {json_path}")
        except Exception as e:
            print(f"저장 실패: {e}")        
        
if __name__ == "__main__" :
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    myWindow = WindowClass()
    myWindow.show()
    app.exec_() 