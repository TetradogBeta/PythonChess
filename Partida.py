import chess;
import chess.svg;
import cairosvg;
from cairosvg import svg2png;
from IPython.display import clear_output;
import cv2;
from google.colab.patches import cv2_imshow;
from cairosvg import svg2png;

class Partida:
    IAModelo=ChessModel;
    DefaultPath='partida';

    def __init__(self,aiWhite=False,newBoard=True,modelo=None):

      if newBoard:
          self.Board=chess.Board();
      else:
          self.Board=None;

      self.IsHumanTurn=not aiWhite;
      self.IsAIWhite=aiWhite;
      self.PlayerMove=Partida._PlayerMove;
      self.Tablas=Partida._Tablas;
      self.Ganas=Partida._Ganas;
      self.Pierdes=Partida._Pierdes;
      self.Pensando=Partida._Pensando;
      if modelo is None:
          modelo=Partida.IAModelo();
      self.Modelo=modelo;

    def Load(self,path=None):
      if path is None:
          path=Partida.DefaultPath;
      else:
          path=str(path);
      #cargo el archivo y leo las variables (Board,IsHumanTurn,IsAIWhite)
    def Save(self,path=None):
      if path is None:
          path=Partida.DefaultPath;
      else:
          path=str(path);
      #guardo en un archivo las variables (Board,IsHumanTurn,IsAIWhite)   
    def Start(self):
      self.play_game();
        
    def play_game(self):
      """Play through the whole game
          
      Keyword arguments:
      turn -- True for A.I plays first
      current_board -- chess.Board()
      """
      if(self.Board.is_stalemate()):
        self.Tablas();

      else:   
        if(self.IsHumanTurn):
          if(not self.Board.is_checkmate()):
            self.PlayerMove(self.Board,self.IsAIWhite);
            self.IsHumanTurn=False;
            self.play_game();
          else:
            self.Pierdes(self.Board,self.IsAIWhite);
                
        else:
          if(not self.Board.is_checkmate()):
            self.Pensando(self.Board,self.IsAIWhite);
            self.Modelo.Move(self.Board,self.IsAIWhite);
            self.IsHumanTurn=True;
            self.play_game()
          else:
            self.Ganas(self.Board,self.IsAIWhite);
                    

    @staticmethod
    def _PlayerMove(current_board,ai_white):
      """Handle the human's turn

      Keyword arguments:
      current_board = chess.Board()
      """
      clear_output()
      Partida._draw_board(current_board,ai_white)
      print('\n')
      print('\n')
      print('ejemplo de uso (a2 a4 ->mueve el primer peon blanco dos celdas adelante) ');
      move_uci =''.join(input('Entre el movimiento: ').split(' ')).lower();
      
      try: 
        move = chess.Move.from_uci(move_uci)
      except:
        return Partida._PlayerMove(current_board) 
      if(move not in current_board.legal_moves):
        return Partida._PlayerMove(current_board)
      current_board.push(move)
      
    
    @staticmethod
    def _Tablas():
      clear_output()
      print('Tablas: A.I y tu GANAIS!')
    @staticmethod
    def _Ganas(current_board,ai_white):
      clear_output()
      Partida._draw_board(current_board,ai_white)
      print('Has GANADO!!')
    @staticmethod
    def _Pierdes(current_board,ai_white):
      clear_output()
      Partida._draw_board(current_board,ai_white)
      print('A.I Gana')
    @staticmethod
    def _Pensando(current_board,ai_white):
      clear_output()
      Partida._draw_board(current_board,ai_white)
      print('\n')
      print(r"""                                        
                          ._ o o
                          \_`-)|_       Espera,  
                      ,""       \        Estoy pensando...
                      ,"  ## |   ಠ ಠ.        
                  ," ##   ,-\__    `.
                  ,"       /     `--._;)
              ,"     ## /
              ,"   ##    /
              """);
    @staticmethod
    def _draw_board(current_board,ai_white):
      """Draw board

      Keyword arguments:
      current_board -- chess.Board()
      """
      board_img = chess.svg.board(current_board, flipped = ai_white)
      svg2png(bytestring=board_img,write_to='/content/board.png')
      img = cv2.imread('/content/board.png', 1)
      cv2_imshow(img)