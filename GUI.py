from tkinter import *
from tkinter import ttk
from Refinement import principal_component_analysis as pca
from tkinter import filedialog
from Testing import landmark_detection

# La clase 'Aplicacion' ha crecido. En el ejemplo se incluyen
# nuevos widgets en el método constructor __init__(): Uno de
# ellos es el botón 'Info'  que cuando sea presionado llamará
# al método 'verinfo' para mostrar información en el otro
# widget, una caja de texto: un evento ejecuta una acción:

class Aplicacion():
    def __init__(self):
        # En el ejemplo se utiliza el prefijo 'self' para
        # declarar algunas variables asociadas al objeto
        # ('mi_app')  de la clase 'Aplicacion'. Su uso es
        # imprescindible para que se pueda acceder a sus
        # valores desde otros métodos:

        self.raiz = Tk()
        self.raiz.geometry('600x400')

        # Impide que los bordes puedan desplazarse para
        # ampliar o reducir el tamaño de la ventana 'self.raiz':

        self.raiz.resizable(width=False, height=False)
        self.raiz.title('Ver info')

        # Define el widget Text 'self.tinfo ' en el que se
        # pueden introducir varias líneas de texto:

        self.tinfo = Text(self.raiz, width=80, height=20)

        # Sitúa la caja de texto 'self.tinfo' en la parte
        # superior de la ventana 'self.raiz':

        self.tinfo.pack(side=TOP)

        # Define el widget Button 'self.binfo' que llamará
        # al metodo 'self.verinfo' cuando sea presionado

        self.binfo = ttk.Button(self.raiz, text='Info',
                                command=self.verinfo)

        # Coloca el botón 'self.binfo' debajo y a la izquierda
        # del widget anterior

        self.binfo.pack(side=LEFT)

        self.bopen = ttk.Button(self.raiz, text='Open File', command=self.open_image)
        self.bopen.pack()

        # Define el botón 'self.bsalir'. En este caso
        # cuando sea presionado, el método destruirá o
        # terminará la aplicación-ventana 'self.raíz' con
        # 'self.raiz.destroy'

        self.bsalir = ttk.Button(self.raiz, text='Salir',
                                 command=self.raiz.destroy)

        # Coloca el botón 'self.bsalir' a la derecha del
        # objeto anterior.

        self.bsalir.pack(side=RIGHT)

        # El foco de la aplicación se sitúa en el botón
        # 'self.binfo' resaltando su borde. Si se presiona
        # la barra espaciadora el botón que tiene el foco
        # será pulsado. El foco puede cambiar de un widget
        # a otro con la tecla tabulador [tab]

        self.binfo.focus_set()
        self.raiz.mainloop()

    def verinfo(self):
        # Borra el contenido que tenga en un momento dado
        # la caja de texto

        self.tinfo.delete("1.0", END)

        # Obtiene información de la ventana 'self.raiz':

        info1 = self.raiz.winfo_class()
        info2 = self.raiz.winfo_geometry()
        info3 = str(self.raiz.winfo_width())
        info4 = str(self.raiz.winfo_height())
        info5 = str(self.raiz.winfo_rootx())
        info6 = str(self.raiz.winfo_rooty())
        info7 = str(self.raiz.winfo_id())
        info8 = self.raiz.winfo_name()
        info9 = self.raiz.winfo_manager()

        # Construye una cadena de texto con toda la
        # información obtenida:

        texto_info = "Clase de 'raiz': " + info1 + "\n"
        texto_info += "Resolución y posición: " + info2 + "\n"
        texto_info += "Anchura ventana: " + info3 + "\n"
        texto_info += "Altura ventana: " + info4 + "\n"
        texto_info += "Pos. Ventana X: " + info5 + "\n"
        texto_info += "Pos. Ventana Y: " + info6 + "\n"
        texto_info += "Id. de 'raiz': " + info7 + "\n"
        texto_info += "Nombre objeto: " + info8 + "\n"
        texto_info += "Gestor ventanas: " + info9 + "\n"

        # Inserta la información en la caja de texto:

        self.tinfo.insert("1.0", texto_info)
        pca.run()

    def open_image(self):
        image_path = filedialog.askopenfilename(title='Open Image',
                                                initialdir='/home/rcruz/workspace/tesis/giancarlos/displasia/detection/images',
                                                filetypes=(('JPG Images', '*.JPG'),('All Files', '*.*')))
        self.tinfo.delete('1.0', END)
        self.tinfo.insert('1.0', 'Testing Image...'+image_path)

        landmark_detection.run(image_path)



def main():
    mi_app = Aplicacion()
    return 0


if __name__ == '__main__':
    main()