package Controle;

import javax.swing.JButton;
import javax.swing.JTextArea;

/**
 *
 * @author Jônatas Trabuco Belotti [jonatas.t.belotti@hotmail.com]
 */
public abstract class Comunicador {
  private static JTextArea jTxtLog = null;
  private static JButton jBtnTestar = null;
  private static JButton jBtnSalvar = null;

  public static void setCampo(JTextArea campo) {
    jTxtLog = campo;
  }
  
  public static void setBotaoTestar(JButton botao) {
    jBtnTestar = botao;
  }

  public static void setjBtnSalvar(JButton botao) {
    jBtnSalvar = botao;
  }
  
  public static void iniciarLog(String texto){
    if (jTxtLog != null) {
      jTxtLog.setText(texto);
    }
  }
  
  public static void addLog(String texto) {
    if (jTxtLog != null) {
      jTxtLog.append("\n" + texto);
      jTxtLog.setCaretPosition(jTxtLog.getText().length());
    }
  }
  
  public static void setEnabledBotaoTestar(boolean valor) {
    if (jBtnTestar != null) {
      jBtnTestar.setEnabled(valor);
    }
  }
  
  public static void setEnabledBotaoSalvar(boolean valor) {
    if (jBtnSalvar != null) {
      jBtnSalvar.setEnabled(valor);
    }
  }

  public static void limparLog() {
    jTxtLog.setText("");
  }

}
