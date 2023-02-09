import sys
import os
from PyQt5.QtWidgets import *
from PyQt5 import uic

UI = 'webcam.ui'

class Dialog(QDialog):
	def __init__(self):
		super().__init__()
		uic.loadUi(UI, self)
		self.setting()
		
	def setting(self):
		self.webcam.clicked.connect(self.webcam0)
	
	def webcam0(self):
		os.system("python3 webcam.py")

app = QApplication(sys.argv)
ex = Dialog()
ex.show()
sys.exit(app.exec_())

