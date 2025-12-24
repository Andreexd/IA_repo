import pygame
from queue import PriorityQueue
import math
from math import inf

ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("A* Visualización")

BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
NARANJA = (255, 165, 0)
PURPURA = (128, 0, 128)
AZUL = (0, 0, 255)
CIAN = (0, 255, 255)

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = col * ancho
        self.y = fila * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def hacer_abierto(self):
        self.color = VERDE

    def hacer_cerrado(self):
        self.color = ROJO

    def hacer_camino(self):
        self.color = AZUL

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)
    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    x, y = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col

def obtener_vecinos(nodo, grid):
    vecinos = []
    filas = nodo.total_filas
    fila, col = nodo.get_pos()
    
    movimientos = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2))
    ]
    
    for df, dc, costo in movimientos:
        nf, nc = fila + df, col + dc
        
        if 0 <= nf < filas and 0 <= nc < filas:
            vecino = grid[nf][nc]
            if not vecino.es_pared():
                if abs(df) == 1 and abs(dc) == 1:
                    if not grid[fila + df][col].es_pared() or not grid[fila][col + dc].es_pared():
                        vecinos.append((vecino, costo))
                else:
                    vecinos.append((vecino, costo))
    
    return vecinos

def heuristica(a_pos, b_pos):
    (x1, y1) = a_pos
    (x2, y2) = b_pos
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def reconstruir_camino(came_from, current_pos, grid, inicio_pos):
    path = []
    while current_pos in came_from:
        path.append(current_pos)
        current_pos = came_from[current_pos]
    for pos in path:
        if pos != inicio_pos:
            fila, col = pos
            grid[fila][col].hacer_camino()
    return list(reversed(path))

def a_star(draw, grid, inicio, fin):
    start_pos = inicio.get_pos()
    end_pos = fin.get_pos()

    contador = 0
    cola_abierta = PriorityQueue()
    
    epsilon = 2.0
    g_score = { (r, c): inf for r in range(inicio.total_filas) for c in range(inicio.total_filas) }
    g_score[start_pos] = 0

    f_score = { (r, c): inf for r in range(inicio.total_filas) for c in range(inicio.total_filas) }
    f_score[start_pos] = g_score[start_pos] + epsilon * heuristica(start_pos, end_pos)

    came_from = {}

    cola_abierta.put((f_score[start_pos], contador, start_pos))
    conjunto_abierto = {start_pos}
    conjunto_cerrado = set()

    while not cola_abierta.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, conjunto_cerrado
                
        current = cola_abierta.get()[2]
        
        if current in conjunto_cerrado:
            continue

        if current in conjunto_abierto:
            conjunto_abierto.remove(current)
        
        conjunto_cerrado.add(current)
        
        if current != start_pos and current != end_pos:
            r, c = current
            grid[r][c].hacer_cerrado()

        draw()
        pygame.time.wait(50)  

        if current == end_pos:
            path = reconstruir_camino(came_from, current, grid, start_pos)
            fin.hacer_fin()
            inicio.hacer_inicio()
            return path, conjunto_cerrado

        cr, cc = current
        current_node = grid[cr][cc]
        vecinos = obtener_vecinos(current_node, grid)
        
        for vecino, costo_movimiento in vecinos:
            vecino_pos = vecino.get_pos()
            
            if vecino_pos in conjunto_cerrado:
                continue
                
            tentative_g = g_score[current] + costo_movimiento

            if tentative_g < g_score[vecino_pos]:
                came_from[vecino_pos] = current
                g_score[vecino_pos] = tentative_g
                f_score[vecino_pos] = tentative_g + epsilon * heuristica(vecino_pos, end_pos)
                
                contador += 1
                cola_abierta.put((f_score[vecino_pos], contador, vecino_pos))
                
                if vecino_pos not in conjunto_abierto:
                    conjunto_abierto.add(vecino_pos)
                    if vecino_pos != end_pos:
                        vecino.hacer_abierto()

    return None, conjunto_cerrado



def main(ventana, ancho):
    FILAS = 11
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None

    corriendo = True
    buscando = False

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                if fila < FILAS and col < FILAS:
                    nodo = grid[fila][col]
                    if not inicio and nodo != fin:
                        inicio = nodo
                        inicio.hacer_inicio()
                    elif not fin and nodo != inicio:
                        fin = nodo
                        fin.hacer_fin()
                    elif nodo != fin and nodo != inicio:
                        nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                if fila < FILAS and col < FILAS:
                    nodo = grid[fila][col]
                    nodo.restablecer()
                    if inicio and nodo.get_pos() == inicio.get_pos():
                        inicio = None
                    elif fin and nodo.get_pos() == fin.get_pos():
                        fin = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin and not buscando:
                    buscando = True
                    for fila in grid:
                        for nodo in fila:
                            if not nodo.es_pared() and not nodo.es_inicio() and not nodo.es_fin():
                                nodo.restablecer()

                    def draw():
                        dibujar(ventana, grid, FILAS, ancho)

                    resultado = a_star(draw, grid, inicio, fin)
                    if resultado:
                        path, closed_set = resultado
                    else:
                        pass

                    buscando = False

                if event.key == pygame.K_c:
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS, ancho)
                
                if event.key == pygame.K_ESCAPE:
                    corriendo = False

    pygame.quit()

if __name__ == "__main__":
    pygame.init()
    main(VENTANA, ANCHO_VENTANA)
