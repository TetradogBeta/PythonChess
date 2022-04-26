from Partida import Partida;

partida=Partida('n' in input('¿Quieres ser las blancas? [s/n]: ').lower());
partida.Load();
partida.Start();