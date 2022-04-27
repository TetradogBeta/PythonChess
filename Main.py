from Partida import Partida;

isDebbuguing=False;
isWhite= isDebbuguing or 'n' in input('¿Quieres ser las blancas? [s/n]: ').lower();
partida=Partida(isWhite);
partida.Load();
partida.Start();