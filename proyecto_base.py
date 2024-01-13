


def scaleSpace(input_image, noct=5, nspo=3, pixel_distance_min=0.5, sigma_min=0.8, sigma_input=0.5):

    ddepth = cv2.CV_64F


    # Redimensiona la imagen de entrada a 256x256 píxeles en nuestro caso.
    im = cv2.resize(input_image, (256, 256), cv2.INTER_LINEAR)

    # Define el número de columnas en la matriz de escalas como el número de escalas por octava más tres.
    column = nspo + 3

    # Inicializa las variables para almacenar los valores de sigma y las máscaras Gaussianas.
    rho = np.zeros(column - 1)
    maskGaussian = []

    # Calcula sigma_zero usando los parámetros proporcionados.
    sigma_zero = sigma_min / pixel_distance_min

    # Calcula los valores de sigma y crea las máscaras Gaussianas correspondientes para cada escala.
    for s in range(column - 1):
        rho[s] = sigma_zero * mt.sqrt(2 ** ((2 * (s + 1)) / nspo) - 2 ** ((2 * s) / nspo))
        maskGaussian.append(gaussianMask1D(rho[s], 0, order=0))

    # Inicializa el espacio de escalas como una lista vacía.
    scale_space = []

    # Crea la semilla para el espacio de escalas doblando el tamaño de la imagen y aplicando un suavizado Gaussiano.
    seed = cv2.resize(im, (round(im.shape[1] * 2), round(im.shape[0] * 2)), cv2.INTER_LINEAR)
    rho_ini = (1 / pixel_distance_min) * np.sqrt(sigma_min ** 2 - sigma_input ** 2)
    maskG = gaussianMask1D(rho_ini, 0, order=0)
    seed = cv2.sepFilter2D(seed, ddepth, kernelX=np.array(maskG), kernelY=np.array(maskG))

    # Crea la primera octava suavizando la semilla con las máscaras Gaussianas.
    initial_octave = [seed]
    for s in range(column - 1):
        new_im = cv2.sepFilter2D(initial_octave[-1], ddepth, kernelX=np.array(maskGaussian[s]), kernelY=np.array(maskGaussian[s]))
        initial_octave.append(new_im)
    scale_space.append(initial_octave)

    # Crea las octavas restantes reduciendo a la mitad la última imagen de la octava anterior y suavizándola.
    for i in range(noct - 1):
        new_octave = []
        new_im = cv2.resize(scale_space[-1][3], (round(scale_space[-1][3].shape[1] / 2), round(scale_space[-1][3].shape[0] / 2)), cv2.INTER_LINEAR)
        new_octave.append(new_im)

        for s in range(column - 1):
            new_im = cv2.sepFilter2D(new_octave[-1], ddepth, kernelX=np.array(maskGaussian[s]), kernelY=np.array(maskGaussian[s]))
            new_octave.append(new_im)

        scale_space.append(new_octave)

    return scale_space



scale_space = scaleSpace(im)

for i in range(len(scale_space)):
  displayMI_ES(scale_space[i], 'Octava' + str(i+1))



def laplacianSpace(scale_space):

  # Control de errores
  laplacian_space = []

  # Número de octavas en el espacio de escala.
  noct = len(scale_space)

  # Número de columnas en cada octava (escala).
  columns = len(scale_space[0]) - 1

  # Iterar a través de las octavas y calcular las diferencias entre las escalas.
  for i in range(noct):
    laplacian_octave = []

    # Calcular las diferencias para cada columna en la octava.
    for j in range(columns):
      laplacian_octave.append(scale_space[i][j + 1] - scale_space[i][j])

    # Agregar la octava calculada al espacio laplaciano.
    laplacian_space.append(laplacian_octave)

  # Retorna el espacio laplaciano resultante.
  return laplacian_space


# Cálculo del espacio Laplaciano
laplacian_space = laplacianSpace(scale_space)

for i in range(len(laplacian_space)):
  displayMI_ES(laplacian_space[i], 'Octava ' + str(i+1))



import numpy as np
import scipy.ndimage as nd

def LaplacianExtrema(laplacian_space, noct, nspo):
    

    keypoints = []

    # Definimos el entorno donde vamos a usarbuscar cada máximo.
    window = np.ones((3, 3, 3))

    for o in range(noct):
        # Preparamos una matriz 3D (escala, x, y) para la octava actual.
        block_oct_np = np.array(laplacian_space[o][:nspo+1])

        # Aplicamos filtros máximo y mínimo 3D para encontrar máximos y mínimos locales en el espacio de escalas.
        block_max_np = nd.maximum_filter(block_oct_np, footprint=window)
        block_min_np = nd.minimum_filter(block_oct_np, footprint=window)

        # Iteramos solo a través de las escalas intermedias y los píxeles interiores.
        for s in range(1, nspo - 1):
            for m in range(1, block_oct_np.shape[1] - 1):
                for n in range(1, block_oct_np.shape[2] - 1):
                    pixel_val = block_oct_np[s, m, n]

                    # Verificamos si el píxel actual es un máximo o mínimo local.
                    if (pixel_val == block_max_np[s, m, n] or pixel_val == block_min_np[s, m, n]):
                        keypoints.append((o, s, m, n))

    return keypoints


kp = LaplacianExtrema(laplacian_space, len(scale_space), len(scale_space[0]))



def gradient(laplacian_space, o , s , m , n):


  # Calcula las diferencias respecto a s, m y n y las almacena en las variables correspondientes.
  dfs = (laplacian_space[o][s + 1][m, n] - laplacian_space[o][s - 1][m, n]) / 2
  dfm = (laplacian_space[o][s][m + 1, n] - laplacian_space[o][s][m - 1, n]) / 2
  dfn = (laplacian_space[o][s][m, n + 1] - laplacian_space[o][s][m, n - 1]) / 2

  # Devolvemos las componentes del vector gradiente en forma de lista.
  return [dfs, dfm, dfn]

def hessian(laplacian_space, o, s, m, n):


  # Cálculo de las segundas derivadas parciales en las direcciones s, m y n.
  h11 = laplacian_space[o][s + 1][m, n] + laplacian_space[o][s - 1][m, n] - 2 * laplacian_space[o][s][m, n]
  h22 = laplacian_space[o][s][m + 1, n] + laplacian_space[o][s][m - 1, n] - 2 * laplacian_space[o][s][m, n]
  h33 = laplacian_space[o][s][m, n + 1] + laplacian_space[o][s][m, n - 1] - 2 * laplacian_space[o][s][m, n]

  # Cálculo de las segundas derivadas cruzadas.
  h12 = (laplacian_space[o][s + 1][m + 1, n] - laplacian_space[o][s + 1][m - 1, n] - laplacian_space[o][s - 1][m + 1, n] + laplacian_space[o][s - 1][m - 1, n]) / 4
  h13 = (laplacian_space[o][s + 1][m, n + 1] - laplacian_space[o][s + 1][m, n - 1] - laplacian_space[o][s - 1][m, n + 1] + laplacian_space[o][s - 1][m, n - 1]) / 4
  h23 = (laplacian_space[o][s][m + 1, n + 1] - laplacian_space[o][s][m + 1, n - 1] - laplacian_space[o][s][m - 1, n + 1] + laplacian_space[o][s][m - 1, n - 1]) / 4

  # Devolvemos la matriz Hessiana en forma de lista de listas.
  return [[h11, h12, h13],
          [h12, h22, h23],
          [h13, h23, h33]]


def outpoint(laplacian_space, o, s, m, n):

  # Verifica si la escala está fuera de los límites.
  out_scale = len(laplacian_space[o]) < s + 2 or s < 1

  if(len(laplacian_space[o][0]) == 0):
    return True

  # Verifica si la coordenada x está fuera de los límites.
  out_x =  m < 1 or laplacian_space[o][0].shape[0] < m + 1
  # Verifica si la coordenada y está fuera de los límites.
  out_y = n < 1 or laplacian_space[o][1].shape[0] < n + 1

  # Devuelve True si al menos una coordenada está fuera de los límites, False en caso contrario.
  if(out_scale or out_x or out_y):
    return True
  else:
    return False


def posRefinement(laplacian_space,keypoints, nspo=3, dist_min=0.5, sigma_min=0.8):
  
  # Keypoints refinados
  kp_ref = []

  # Factor de upsampling, lo usaremos para calcular la distancia entre pixeles de cada escala
  dist = 2

  # Para cada extremo detectado
  for extrema in keypoints:
    it = 0
    cond = True
    o = extrema[0]
    s = extrema[1]
    m = extrema[2]
    n = extrema[3]

    # Nos aseguramos que no estamos en un borde, pues no podríamos calcular derivadas
    if(not (outpoint(laplacian_space, o, s + 1, m + 1, n + 1)) and not (outpoint(laplacian_space, o, s - 1, m - 1, n - 1))):
      while cond:
        # Calculamos el gradiente en el punto extremo.
        G = gradient(laplacian_space, o, s, m, n)
        # Calculamos la matriz Hessiana en el punto extremo.
        H = hessian(laplacian_space, o, s, m, n)
        # Calculamos el máximo en la función continua.
        aestrella = -np.linalg.inv(H).dot(G)

        # Calculamos el valor del desarrollo de Taylor.
        w = laplacian_space[o][s][m][n]
        omega = w + (aestrella.dot(G)) / 2

        # Calculamos las coordenadas absolutas ajustando la escala, la posición y la distancia entre puntos.
        sigma = ((dist**(o-1)) / dist_min) * sigma_min * 2 ** ((aestrella[0] + s) / nspo)
        x = dist**(o-1) * (aestrella[1] + m)
        y = dist**(o-1) * (aestrella[2] + n)

        # Si la magnitud de la corrección es pequeña, consideramos que hemos encontrado la posición refinada.
        if (np.abs(aestrella).max() < 0.6):
          kp_ref.append([o, s, m, n, sigma, x, y, omega])
          cond = False

        # En caso contrario, ajustamos el nuevo pixel para buscar el refinamiento
        else:
          it += 1
          s = round(extrema[1] + aestrella[0])
          m = round(extrema[2] + aestrella[1])
          n = round(extrema[3] + aestrella[2])

          # Si hemos alcanzado el límite de iteraciones o estamos fuera de los límites, terminamos el refinamiento.
          if (it == 4 or outpoint(laplacian_space, o, s, m, n)):
            cond = False

  return kp_ref



def lowContrastExtremaSupression(laplacian_space, discrete_extrema, CDoG=0.015, factor=0.8):
    

    refinement = []  # Inicializamos la lista de keypoints refinados.

    # Iteramos sobre cada keypoint detectado.
    for extrema in discrete_extrema:
        # Accedemos al valor de Laplacian en la ubicación del keypoint.
        laplacian_value = np.abs(laplacian_space[extrema[0]][extrema[1]][extrema[2], extrema[3]])

        # Comparamos el valor absoluto del Laplacian con el umbral ajustado de contraste.
        if laplacian_value >= factor * CDoG:
            # Si el keypoint tiene suficiente contraste, lo añade a la lista de refinados.
            refinement.append(extrema)

    # Devolvemos la lista de keypoints que han pasado el filtro de contraste.
    return refinement



def edgeExtremaSupression(laplacian_space, keypoints, Cedge=10):
   
    no_edge_extrema = []  # Lista para almacenar keypoints que no están en bordes.

    # Itera sobre cada keypoint.
    for kp in keypoints:
      # COmprobamos Limites
      if(not (outpoint(laplacian_space, kp[0], kp[1], kp[2], kp[3]))):
        # Obtiene la matriz Hessiana en la ubicación del keypoint.
        H = hessian(laplacian_space, kp[0], kp[1], kp[2], kp[3])
        # Extrae los elementos de la matriz Hessiana.
        H_11 = H[1][1]
        H_12 = H[1][2]
        H_22 = H[2][2]

        # Calcula el determinante y la traza de la matriz Hessiana proyectada.
        detH = H_11 * H_22 - H_12 * H_12
        traceH = H_11 + H_22

        # Calcula la medida de "edgeness" para el keypoint.
        # Nota: Se debe verificar la posibilidad de division por cero cuando traceH es muy pequeña??????????????????????????
        edgeness = traceH**2 / detH if detH != 0 else float('inf')

        # Comprueba si el keypoint cumple con el criterio de "edgeness".
        if(edgeness < (Cedge + 1)**2 / Cedge):
            # Si el keypoint no está en un borde, se añade a la lista de keypoints válidos.
            no_edge_extrema.append(kp)

    return no_edge_extrema


# Eliminamos los keypoints con bajo contraste
kp_lowcontrast = lowContrastExtremaSupression(laplacian_space, kp)

print('Keypoints iniciales: ' + str(len(kp)))
print('Keypoints tras refinamiento: ' + str(len(kp_lowcontrast)))
print('-------------------------------------')
print('Keypoints eliminados: ' + str(len(kp) - len(kp_lowcontrast)))

# Eliminamos los keypoints en los bordes
kp_noedge = edgeExtremaSupression(laplacian_space, kp_lowcontrast)

print('Keypoints iniciales: ' + str(len(kp_lowcontrast)))
print('Keypoints tras refinamiento: ' + str(len(kp_noedge)))
print('-------------------------------------')
print('Keypoints eliminados: ' + str(len(kp_lowcontrast) - len(kp_noedge)))

# Finalmente aplicamos el desarrollo de Taylor y refinamos las posiciones
kp_refined = posRefinement(laplacian_space,kp_noedge)

print('Keypoints iniciales: ' + str(len(kp_noedge)))
print('Keypoints tras refinamiento: ' + str(len(kp_refined)))
print('-------------------------------------')
print('Keypoints eliminados: ' + str(len(kp_noedge) - len(kp_refined)))


# Cargamos la imagen en color para poder dibujar sobre ella
im_color = readIm(get_image('IMG_20211030_110410_S.jpg'),1)

fig, ax = plt.subplots()

im_color = cv2.resize(im_color, (256, 256), cv2.INTER_LINEAR)

ax.imshow(im_color)

# Centramos en cada keypoint una circunferencia de radio variable en función de sigma
for i in range(len(kp_refined)):
  x = round(kp_refined[i][5])
  y = round(kp_refined[i][6])
  radius = int(1.5 * kp_refined[i][4])
  circle = plt.Circle((y, x), radius, color='r', fill=False)
  ax.add_patch(circle)

plt.show()



def gradientScaleSpace(scale_space, noct, nspo, mask = [-1/2,0,1/2], vert = True):


    ddepth = cv2.CV_64F
    gradient_space = []  # Inicializa el espacio de escala de gradientes

    # Itera a través de cada octava
    for o in range(noct):
        gradient_octave = []  # Inicializa la lista para los gradientes de la octava actual

        # Itera a través de cada escala dentro de la octava
        for s in range(nspo):
            # Define las máscaras para el filtro separable basado en 'vert'
            maskvert = mask if vert else [0, 1, 0]
            maskhori = [0, 1, 0] if vert else mask

            # Aplica el filtro separable a la imagen en la escala actual
            gradient_octave.append(cv2.sepFilter2D(scale_space[o][s], ddepth, kernelX=np.array(maskhori), kernelY=np.array(maskvert)))

        # Añade los gradientes de la octava actual al espacio de escala de gradientes
        gradient_space.append(gradient_octave)

    # Devuelve el espacio de escala de gradientes completo
    return gradient_space

def orientationHistogramAccumulation(gradient_img_x, gradient_img_y, x_key, y_key, threshold_dist, sigma_key, delta_o, nbins, lambda_ori=1.5):

    # Inicializamos el histograma
    histogram = [0] * nbins

    # En la primera escala no podemos dividir por delta_o porque estamos multiplicando por 2 y nos salimos de la ventana
    #if(delta_o < 1):
    #  delta_o = 1

    # Inicializamos las coordenadas de la ventana
    m_min = round((x_key - threshold_dist) / delta_o)
    m_max = round((x_key + threshold_dist) / delta_o)
    n_min = round((y_key - threshold_dist) / delta_o)
    n_max = round((y_key + threshold_dist) / delta_o)

    #print('Tamaño imagen: ' + str(gradient_img_x.shape[0]) + 'x' + str(gradient_img_x.shape[1]))
    #print('M minimo: ' + str(m_min))
    #print('M maximo: ' + str(m_max))
    #print('N minimo: ' + str(n_min))
    #print('N minimo: ' + str(n_max))
    #print('delta_o: ' + str(delta_o))
    height = gradient_img_x.shape[0]
    width = gradient_img_x.shape[1]

    # Iteramos sobre la ventana de interés
    for m in range(m_min, m_max): # m_min ----> m_max
        for n in range(n_min, n_max): # n_min ----> n_max
          if 0 <= m < height and 0 <= n < width:
            #print('(m,n) = (' + str(m) + ',' + str(n) + ')' )
            # Calculamos la norma del gradiente en el punto (m, n)
            norm = np.sqrt(gradient_img_x[m][n]**2 + gradient_img_y[m][n]**2)
            # Calculamos la distancia ponderada por la función gaussiana
            dist_mn = ((m * delta_o - x_key)**2 + (n * delta_o - y_key)**2)
            c_mn = np.exp(-dist_mn / (2 * (lambda_ori * sigma_key)**2)) * norm

            # Calculamos la orientación correspondiente
            mod = mt.fmod(mt.atan2(gradient_img_y[m][n], gradient_img_x[m][n]), 2 * np.pi)
            b_mn = round((nbins / (2 * np.pi)) * mod)

            # Acumulamos en el histograma
            histogram[b_mn] += c_mn

    return histogram


def smoothHistogram(histogram, mask=[1/3, 1/3, 1/3], ntimes=6):

  padded_histogram = histogram

  for i in range(ntimes):
    # Agrega el primer y último elemento al final y al inicio del histograma, respectivamente, para realizar una convolución circular
    # Es necesario actualizar los bordes en cada iteración para que realmente sea una covolcuión circular
    padded_histogram.append(padded_histogram[0]) # Añadimos el priumer elemento al final de la lista
    padded_histogram.insert(0, padded_histogram[-2]) # En -2 está el útimo elemento de histogram actualizado

    # Aplica la convolución al histograma modificado con la máscara especificada
    # Al aplicar 'valid' se eliminan los bordes
    padded_histogram = np.convolve(padded_histogram, mask, 'valid')

  return padded_histogram



def referenceOrientationExtraction(histogram, nbins, t_value=0.8):


    referenceOrientations = []  # Inicializa la lista para almacenar las orientaciones de referencia
    h_max_index = np.argmax(histogram)  # Encuentra el índice del valor máximo en el histograma
    h_max = histogram[h_max_index]  # Obtiene el valor máximo en el histograma


    # Itera a través de cada bin en el histograma
    for i in range(nbins):
        # Verifica si el bin actual tiene un valor más alto que sus vecinos y está por encima del umbral
        if (histogram[i] > histogram[i - 1]
            and histogram[i] > histogram[(i + 1) % nbins]
            and histogram[i] > t_value * h_max):
            # Calcula la orientación de referencia utilizando la interpolación
            referenceOrientations.append(
                np.pi / nbins * (2 * (i - 1) + (histogram[i - 1] - histogram[(i + 1) % nbins]) / (histogram[i - 1] - 2 * histogram[i] + histogram[(i + 1) % nbins]))
            )

    return referenceOrientations


def mostrarHistograma(data):
  # Creamos un histograma a partir de la lista de datos
  plt.hist(data, bins=10, alpha=0.7, color='blue')

  # Añadimos títulos y etiquetas para mayor claridad
  plt.title('Histograma de la lista de datos')
  plt.xlabel('Valor')
  plt.ylabel('Frecuencia')

  # Mostramos la gráfica
  plt.show()

def keypointReferenceOrientation(scale_space, scale_space_gradient_x, scale_space_gradient_y, keypoints, lambda_ori=1.5, nbins=36):


    reference_orientations_kp = []  # Inicializa la lista para almacenar las orientaciones de referencia
    downsampling_factor = 2  # Factor de muestreo para downsampling

    # Itera a través de cada keypoint
    for kp in keypoints:
        o = kp[0]  # Octava del keypoint
        s = kp[1] - 1  # Escala del keypoint, recordemos que las escalas van de 1 a 3 porque la primera y la ultima no la consideramos
        sigma_key = kp[4]  # Sigma del keypoint
        x_key = kp[5]  # Coordenada x del keypoint
        y_key = kp[6]  # Coordenada y del keypoint

        if(s == 0):
          print('o: ' + str(o))
          print('s: ' + str(s))

        #print('n oct: ' + str(len(scale_space_gradient_x)))
        #print('n spo: ' + str(len(scale_space_gradient_x[0])))

        # Obtiene la imagen, gradientes en x e y para el keypoint actual
        img = scale_space[o][s]
        gradient_img_x = scale_space_gradient_x[o][s]
        gradient_img_y = scale_space_gradient_y[o][s]

        # Dimensiones de la imagen
        height = gradient_img_x.shape[0]
        width = gradient_img_x.shape[1]

        # Distancia de umbral para verificar si el keypoint está lo suficientemente lejos de los bordes
        threshold_dist = 3 * lambda_ori * sigma_key
        delta_o = downsampling_factor ** (o - 1)

        hist_cont = 0

        # Comprobar que el keypoint está lo suficientemente lejos de los bordes
        if (
            (threshold_dist <= x_key and x_key  <= (height - threshold_dist))
            and (threshold_dist <= y_key and y_key <= (width - threshold_dist))
        ):
            # A. HISTOGRAMA DE ORIENTACIÓN ACUMULADA
            histogram = orientationHistogramAccumulation(
                gradient_img_x, gradient_img_y, x_key, y_key, threshold_dist, sigma_key, delta_o, nbins, lambda_ori
            )

            # B. SUAVIZADO DEL HISTOGRAMA
            # HE CAMBIADO EL SMOOTH HISTOGRAMA !!!!!!!!!!!!!!
            histogram = smoothHistogram_corregido(histogram)

            #if(hist_cont < 100):
              #mostrarHistograma(histogram)
              #hist_con = hist_cont + 1

            # C. EXTRACCIÓN DE ORIENTACIONES DE REFERENCIA
            references_orientations = referenceOrientationExtraction(histogram, nbins, 0.8)

            #if(hist_cont < 100):
              #for ori in references_orientations:
                #print(ori)

            # Añadimos las orientaciones de referencia junto con su kp correspondiente a la lista
            for theta_ref in references_orientations:
              reference_orientations_kp.append((o,s,sigma_key,x_key,y_key, theta_ref))

    # Retorna la lista de orientaciones de referencia para todos los keypoints
    return reference_orientations_kp

def gradientScaleSpace_corregido(scale_space, noct, nspo, mask=[-1/2, 0, 1/2], vert=True):
    ddepth = cv2.CV_64F
    gradient_space = []

    for o in range(noct):
        gradient_octave = []

        for s in range(nspo):
            # Convertir la máscara en un array de NumPy y asegurarse de que es un vector fila
            mask_np = np.array(mask, dtype=float).reshape(1, -1)

            # Crear kernels como vectores columna o fila
            if vert:
                maskvert = mask_np.T  # Transponer para obtener un vector columna
                maskhori = np.array([[0, 1, 0]], dtype=float)
            else:
                maskvert = np.array([[0], [1], [0]], dtype=float)
                maskhori = mask_np  # Usar como vector fila

            # Aplicar el filtro separable
            gradient_octave.append(cv2.sepFilter2D(scale_space[o][s], ddepth, kernelX=maskhori, kernelY=maskvert))

        gradient_space.append(gradient_octave)

    return gradient_space

def smoothHistogram_corregido(histogram, mask=[1/3, 1/3, 1/3], ntimes=6):
    # ESTO Lastra decía que era el mayor crimen que se podía cometer programando (Sé que eres Nasr)
    if isinstance(histogram, list):
        histogram = np.array(histogram)  # Convierte el histograma en un array de NumPy si es una lista

    for i in range(ntimes):
        # Añade el primer y último elemento al final y al inicio del histograma para una convolución circular
        padded_histogram = np.pad(histogram, (1, 1), mode='wrap')

        # Aplica la convolución al histograma modificado con la máscara especificada
        histogram = np.convolve(padded_histogram, mask, 'valid')

    return histogram

type(scale_space), type(scale_space[0]), len(scale_space), len(scale_space[0])

scale_space_gradient_x = gradientScaleSpace_corregido(scale_space, 5, 3, [-1/2,0,1/2],False)
scale_space_gradient_y = gradientScaleSpace_corregido(scale_space, 5, 3, [-1/2,0,1/2],True)

kp_orientations =  keypointReferenceOrientation(scale_space, scale_space_gradient_x, scale_space_gradient_y, kp_refined, 1.5, 36)



def keypointDescriptor(scale_space, scale_space_gradient_x, scale_space_gradient_y, keypoints, lambda_descr = 6):
   
    nhist = 4 # valor por defecto
    nori = 8  # valor por defecto
    t = 0.8           # threshold estándar a la hora de buscar una segunda orientación de referencia
    downsampling_factor = 2  # Factor de muestreo para downsampling

    descriptor = []

    for kp in keypoints:
      feature_vector = []

      o = kp[0]  # Octava del keypoint
      s = kp[1]  # Escala del keypoint
      sigma_key = kp[2]  # Sigma del keypoint
      x_key = kp[3]  # Coordenada x del keypoint
      y_key = kp[4]  # Coordenada y del keypoint
      theta = kp[5]   # Orientación preferente del keypoint

      # Obtiene la imagen, gradientes en x e y para el keypoint actual
      img = scale_space[o][s]
      gradient_img_x = scale_space_gradient_x[o][s]
      gradient_img_y = scale_space_gradient_y[o][s]

      # Dimensiones de la imagen
      height = gradient_img_x.shape[0]
      width = gradient_img_x.shape[1]

      # Distancia de umbral para verificar si el keypoint está lo suficientemente lejos de los bordes
      threshold_dist = np.sqrt(2) * lambda_descr * sigma_key
      delta_o = downsampling_factor ** (o - 1)
      #if(delta_o < 1):
      #  delta_o = 1

      if (
            (threshold_dist <= x_key <= (height - threshold_dist))
            and (threshold_dist <= y_key <= (width - threshold_dist))
        ):

          # Inicializamos el histograma (i,j)
          histogram = [[[0 for _ in range(nori)] for _ in range(nhist)] for _ in range(nhist)]
          # manera alternativa por si esto no va np.zeros((nhist, nhist, nori))

          c = (nhist+1)/nhist
          # Inicializamos las coordenadas de la ventana
          m_min = round((x_key - threshold_dist*c) / delta_o)
          m_max = round((x_key + threshold_dist*c) / delta_o)
          n_min = round((y_key - threshold_dist*c) / delta_o)
          n_max = round((y_key + threshold_dist*c) / delta_o)

          # Iteramos sobre la ventana de interés
          for m in range(m_min, m_max): # m_min ----> m_max
            for n in range(n_min, n_max): # n_min ----> n_max
              #COMPROBACION QUE NO SE SI DEBERÍAMOS DE PONER
              if 0 <= m < height and 0 <= n < width:
                # Calculamos las coordenadas normalizadas
                x_norm = ((m*delta_o - x_key)*mt.cos(theta) + (n*delta_o - y_key)*mt.sin(theta))/sigma_key
                y_norm = (-(m*delta_o - x_key)*mt.sin(theta) + (n*delta_o - y_key)*mt.cos(theta))/sigma_key

                # Comprobamos que la muetra (m,n) se encuentra dentro de la ventana normalizada
                if(max(abs(x_norm), abs(y_norm)) < lambda_descr*c):
                  # Calculamos la orientación del gradiente normalizada
                  theta_norm = mt.fmod(mt.atan2(gradient_img_y[m][n], gradient_img_x[m][n]) - theta, 2 * np.pi)

                  # Calculamos la contribución
                  norm = np.sqrt(gradient_img_x[m][n]**2 + gradient_img_y[m][n]**2)
                  dist_mn = ((m * delta_o - x_key)**2 + (n * delta_o - y_key)**2)
                  c_mn_d = np.exp(-dist_mn / (2 * (lambda_descr * sigma_key)**2)) * norm

                  for i in range(nhist):
                    for j in range(nhist):
                      x_norm_i = (i - (1+nhist)/2)*(2*lambda_descr / nhist)
                      y_norm_j = (j - (1+nhist)/2)*(2*lambda_descr / nhist)
                      if( abs(x_norm_i - x_norm) <= ((2*lambda_descr)/nhist) and abs(y_norm_j - y_norm) <= ((2*lambda_descr)/nhist)):
                        for k in range(nori):
                          theta_norm_k = 2*np.pi * (k-1) / nori
                          if( abs(mt.fmod(theta_norm_k - theta_norm, 2 * np.pi)) <= ((2 * np.pi)/nori) ):
                            histogram[i][j][k] = histogram[i][j][k] + (1 - (nhist/2*lambda_descr)*abs(x_norm - x_norm_i))*(1 - (nhist/2*lambda_descr)*abs(y_norm - y_norm_j))*( 1 - (nori/2*np.pi)*abs( mt.fmod(theta_norm - theta_norm_k, 2*np.pi) ) )*c_mn_d

          histogram_array = np.array(histogram)
          #feature_vector = histogram_array[1:-1, 1:-1, :].flatten()
          feature_vector = histogram_array.flatten()

          for l in range(nhist*nhist*nori):
            f_norm = np.linalg.norm(feature_vector)
            feature_vector[l] = min(feature_vector[l], 0.2*f_norm)
            if f_norm != 0:
              feature_vector /= np.linalg.norm(feature_vector)
              feature_vector = (512 * feature_vector).clip(0, 255).round()
            feature_vector[feature_vector < 0] = 0
            feature_vector[feature_vector > 255] = 255

          descriptor.append((o, s, x_key, y_key, sigma_key, theta, feature_vector))

    return descriptor

descriptors= keypointDescriptor(scale_space, scale_space_gradient_x, scale_space_gradient_y,kp_orientations , 6)

len(kp_refined), len(descriptors)



def detectandcompute(im, noct=5, nspo=3):
  scale_space = scaleSpace(im)
  laplacian_space = laplacianSpace(scale_space)
  kp = LaplacianExtrema(laplacian_space, len(scale_space), len(scale_space[0]))
  kp_lowcontrast = lowContrastExtremaSupression(laplacian_space, kp)
  kp_noedge = edgeExtremaSupression(laplacian_space, kp_lowcontrast)
  kp_refined = posRefinement(laplacian_space,kp_noedge)

  scale_space_gradient_x = gradientScaleSpace_corregido(scale_space, noct, nspo, [-1/2,0,1/2],False)
  scale_space_gradient_y = gradientScaleSpace_corregido(scale_space, noct, nspo, [-1/2,0,1/2],True)

  kp_orientations = keypointReferenceOrientation(scale_space, scale_space_gradient_x, scale_space_gradient_y, kp_refined, 1.5, 36)
  print(len(kp_orientations))
  kp_descriptor = keypointDescriptor(scale_space, scale_space_gradient_x, scale_space_gradient_y, kp_orientations, 6)

  final_kp = [cv2.KeyPoint(float(i[2]), float(i[3]), i[4], i[5]) for i in kp_descriptor]
  final_desc = [i[6] for i in kp_descriptor]


  return final_kp, final_desc

final_kp, final_descr = detectandcompute(im)

len(final_kp),  len(final_descr)

sift = cv2.SIFT_create()
libreria_kp, libreria_descr = sift.detectAndCompute(im,None)

len(libreria_kp), len(libreria_descr)

print(libreria_kp[0]), print(final_kp[0]), print(final_descr[20]), print(libreria_descr[9])

def siftPoints(im,nfeatures=5000):
  # To be completed by the students
  sift = cv2.SIFT_create()
  kp, ds = sift.detectAndCompute(im,None)

  kp = sorted(kp, key=lambda x: -x.response)
  if len(kp) > nfeatures:
    kp = kp[:nfeatures]
  kp, ds = sift.compute(im, kp)

  return kp,ds

comparar_l_sift, _ = siftPoints(im, 235)

def showKP(im,kp,title):
  img=cv2.drawKeypoints(im, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  displayIm(img,title,1)

print(kp_refined)

showKP(im, comparar_l_sift, 'SIFT Libreria')
showKP(im, final_kp, 'SIFT implementado')
