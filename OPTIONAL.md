# Cambios en la nueva versión del bot

Este documento presenta todas las modificaciones hechas al bot. Junto al documento `advanced_ml_bot.py` que es todas las dependencias de código
autocontenidas en un fichero existe la carpeta `advanced_ml_files` con la informacion persistente que necesita para funcionar. Deben estar en la misma
carpeta.

También, funcionan con dos asunciones:

1. La estructura del motor es la misma: 
	
	- from adversarial_search.chess_game_state import ChessGameState
	- ...
	
Esto es necesario para la clase bot registry que busca por clases compartidas. Una implementación totalmente autocontenida no es detectada.
	
2. Como la estructura es la misma, que todos estos ficheros estan en la carpeta bots. Esto es para rutas de ficheros.

## 1. Nuevo bot
- Se ha creado un nuevo bot que hereda de `MLBot`, pero implementa la búsqueda con alpha/beta y sus alteraciones de prácticas pasadas:
  - Killer moves
  - Tablas de transposición
  - Cutoff test con tiempo implementado

### y, las mas importantes:
- El uso del algoritmo implementado en la primera práctica: `iterative_deepening`.
- Se ha trabajado en la lógica para alternar entre:
  - Posiciones de apertura
  - Algoritmos de búsqueda
  - Endgame

## 2. Base de datos de apertura y endgame
- Se ha añadido una BBDD en formato JSON con posiciones de apertura. Se carga a memoria directamente como un diccionario.
- Para el cutoff test, se guarda una variable, en el bot, con el tiempo permitido por turno. Inicialmente es `null` mientras se encuentren movimientos en el diccionario, funcionando como un booleano.
- Una vez tenga un valor asignado sabemos que nos hemos salido de las lineas almacenadas e ignoramos cualquier búsqueda de apertura.
- Se ha intentado hacer algo similar para el **endgame**:
  - `Chess.board` tiene métodos que devuelven el número de piezas disponibles.
  - Actualmente, toda posición con menos de 7 piezas es resoluble.
  - Bases de datos de Terabytes son inviables, por lo que una implementación con 4-5 piezas parecia mas razonable (1GB).
  - Las bases Syzygy actuales no proveen la métrica necesaria (WDL y DTZ), necesitamos DTM (distance to mate).
	- WDL solo nos dice si la posicion PUEDE, JUGANDO PERFECTAMENTE, acabar en victoria derrota o empate.
	- Si los movimientos no son perfectos, viniendo de la BBDD, guiarnos por esta informacion puede ser contraproducente.
	- DTZ es para la regla de 50 movimientos.
  - Existen bases antiguas como Nalimov, pero no las conseguí y son formatos que no funcionan con el motor de chess.
  
- Por estas dificultades es que se abandono esta idea. Realmente solo faltaba encontrar la BBDD porque la lógica se implemento y era la siguiente:

  - Dada una BBDD con DTM, buscar a profundidad uno el mejor movimiento, es decir, el que mejor resultado de (WDL) y, de ellos, el que mejor numero de turnos nos devuelve.
  - Mas turnos para derrotas, menos para victorias.


## 3. Cambios en la clase `MLMODEL`

### Con las sugerencias de la práctica:
- Multiplicar correctamente para pasar a centipawns (1267 y no 20000).
- Abandonar el aspecto 3D (bitboard) de la entrada.
- Para compensar esta "perdida" de informacion se ha ampliado el **vector de características**, incluyendo información de:
  1. Turno: +1.0 si juega blancas, -1.0 si juega negras
  2. Fase de la partida:
     - Fase (material total normalizado)
     - Complemento de fase (1.0 - fase)
  3. Material:
     - Material blanco normalizado
     - Material negro normalizado
     - Diferencia de material normalizada
  4. Conteo de piezas blancas (normalizado):
     - Peones: /8
     - Caballos: /2
     - Alfiles: /2
     - Torres: /2
     - Dama: /1
  5. Conteo de piezas negras (igual)
  6. Estructura de peones:
     - Avance medio de peones blancos y negros
     - Media del rank de los peones (0–1)
     - Número de peones blancos y negros
  7. Control del centro:
     - Piezas en {D4, D4, E4, E5} / 4
  8. Actividad del rey:
     - Distancia al centro del tablero
  9. Estado táctico:
     - Rey blanco en jaque: 1.0 si sí, 0.0 si no
     - Rey negro en jaque: igual
  10. Derechos de enroque:
      - Blanco/Negro puede enrocar corto
  11. Movilidad:
      - Número de movimientos legales / 50.0


Toda normalizada.

### Intentos de modelos
- **Red neuronal**: Input -> Dense(32) -> Dense(16) -> Output  
  - Muy precisa, pero lenta (40s). El overhead de llamada parece ser el problema, no las capas. Es ya MUY sencillo, visto esto se abandono la idea de una NN.
  - Destacar lo precisa que llega a ser, tanto como la implementacion de bitboards. Lo que me hace dar cuenta de lo poco que realmente aprendío esta.
- **Regresor lineal**:
  - Pipeline:
    ```python
    linreg = Pipeline([
        ("scaler", StandardScaler()),  # Los datos ya deberían estar escalados
        ("lr", Ridge(alpha=10.0, fit_intercept=True, solver="auto", random_state=42))
    ])
    ```
  - Sencillo y rápido, pero pierde contra `H3` de la primera práctica.
  - Con más tiempo, una buena extracción de características + regresor lineal podría ser un baseline sólido, mejor que heurísticas manuales.
  - No estoy seguro que rompe al regresor lineal. Entiendo, visto la precision de la NN, que las caracteristicas son sólidas.
  - L2 debería ayudar con la dimensionalidad del vector, diferentes valores alpha no cambiaban el resultado.
  - Lo logico seria mantener H3 pero he querido, y he visto mas logico, mantener la parte correspondiente a la última práctica.
  


## 4. Reordenación MVV
- Se probó como primera idea de reordenacion simplemente porque fue una recomendación que diste en clase.
- Si hubiese ayudado se hubiesen probado otros patrones vistos. 
- Resultado: no ahorraba tiempo.
  - Probe en diferentes escenarios (partidas mas avanzadas) donde deberian darse mas situacion donde funcione para bien.
  - Por eso, se retiró esta funcionalidad.
  - Por ejemplo, los siguentes resultados son movimientos identicos, con una diferencia clara por el coste de reordenacion:

---

- Total moves: 26  
  Game duration: 74.1s  
  White time left: 179.4s  
  Black time left: 158.6s
  

- Total moves: 26  
  Game duration: 83.8s  
  White time left: 169.8s  
  Black time left: 158.4s


- El bot juega de blanco, la primera partida es sin reordenar.