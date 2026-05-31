import pandas as pd
import numpy as np

# Parâmetros físicos do modelo SDOF (seção 12.2.2)
m = 32500
k = 1908100
zeta = 0.10
c = 2 * zeta * np.sqrt(k * m)
F0 = 22800
omega = 6
G = 9.81  # m/s²


def amplitude_edo():
    """Amplitude teórica em regime permanente na frequência de operação (seção 6.2)."""
    return F0 / np.sqrt((k - m * omega ** 2) ** 2 + (c * omega) ** 2)


def amplitude_aceleracao_rms_ressonancia():
    """Aceleração RMS teórica na ressonância (ω = ωₙ), em múltiplos de g.

    Na ressonância, o termo (k − m·ωₙ²) → 0 e a amplitude de deslocamento atinge
    o máximo teórico A_max = F₀ / (c·ωₙ). A aceleração de pico é ωₙ²·A_max e seu
    valor RMS é (ωₙ²·A_max) / √2. É a referência física para o gatilho absoluto.
    """
    omega_n = np.sqrt(k / m)
    A_max = F0 / (c * omega_n)
    a_pico = omega_n ** 2 * A_max
    a_rms = a_pico / np.sqrt(2)
    return a_rms / G


# Limites operacionais
LIMITE_VARIACAO_REL = 0.30   # 30 % de variação relativa do RMS entre pontos
LIMITE_RMS_PONTO_G = 0.40    # RMS médio do ponto considerado alto (defeito da via)

# Limite absoluto por vagão: derivado da EDO da seção 6.2.
# Considera-se ALERTA quando o RMS observado atinge 50 % da aceleração RMS
# teórica que o sistema produziria operando na frequência natural ωₙ.
FATOR_LIMITE_RESSONANCIA = 0.50
LIMITE_RMS_VAGAO_G = FATOR_LIMITE_RESSONANCIA * amplitude_aceleracao_rms_ressonancia()


def descritores_por_passagem(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa a série temporal por (vagao_id, ponto_id) e calcula RMS e pico."""
    return (
        df.groupby(['vagao_id', 'ponto_id'])
          .agg(
              pico_g=('accel_z_g', lambda s: float(np.max(np.abs(s)))),
              rms_g=('accel_z_g', lambda s: float(np.sqrt(np.mean(s ** 2)))),
          )
          .reset_index()
    )


def avaliar_vagoes(desc: pd.DataFrame) -> pd.DataFrame:
    """Avalia cada vagão por DOIS gatilhos independentes:

    1. **Gatilho relativo**: variação > LIMITE_VARIACAO_REL entre pontos da malha
       → captura defeitos LOCALIZADOS (uma roda com flat spot, p.ex.).
    2. **Gatilho absoluto**: RMS máximo > LIMITE_RMS_VAGAO_G (derivado da EDO)
       → captura proximidade da ressonância ou desgaste GENERALIZADO; o limite
       é 50 % da aceleração RMS teórica do sistema em ω = ωₙ (seção 6.2).

    Ambos os critérios são complementares: o primeiro nunca detectaria um vagão
    operando em ressonância (vibração alta em todos os pontos, variação ≈ 0); o
    segundo nunca detectaria um defeito de roda esporádico (RMS médio normal).
    """
    A_ref_mm = 1000 * amplitude_edo()
    relatorio = []
    for vagao, g in desc.groupby('vagao_id'):
        rms = g['rms_g'].values
        if len(rms) < 2:
            continue
        media = float(np.mean(rms))
        rms_max = float(np.max(rms))
        rms_min = float(np.min(rms))
        variacao_rel = (rms_max - rms_min) / media

        alerta_variacao = variacao_rel > LIMITE_VARIACAO_REL
        alerta_ressonancia = rms_max > LIMITE_RMS_VAGAO_G

        if alerta_variacao and alerta_ressonancia:
            status = 'ALERTA_DUPLO'
        elif alerta_ressonancia:
            status = 'ALERTA_RESSONANCIA'
        elif alerta_variacao:
            status = 'ALERTA_VARIACAO'
        else:
            status = 'NORMAL'

        relatorio.append({
            'vagao_id': vagao,
            'pontos': int(len(rms)),
            'rms_min_g': rms_min,
            'rms_max_g': rms_max,
            'variacao_rel': variacao_rel,
            'A_edo_mm': A_ref_mm,
            'lim_rms_vagao_g': LIMITE_RMS_VAGAO_G,
            'status': status,
        })
    return pd.DataFrame(relatorio)


def avaliar_pontos(desc: pd.DataFrame) -> pd.DataFrame:
    """Para cada ponto, verifica se vários vagões registram RMS alto — indício
    de defeito da via (e não de um vagão isolado)."""
    relatorio = []
    for ponto, g in desc.groupby('ponto_id'):
        rms = g['rms_g'].values
        rms_medio = float(np.mean(rms))
        n_alto = int(np.sum(rms > LIMITE_RMS_PONTO_G))
        status = 'ALERTA_VIA' if n_alto >= 2 else 'NORMAL'
        relatorio.append({
            'ponto_id': ponto,
            'vagoes_observados': int(len(rms)),
            'rms_medio_g': rms_medio,
            'vagoes_acima_limite': n_alto,
            'status': status,
        })
    return pd.DataFrame(relatorio)


def analisar_malha(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    desc = descritores_por_passagem(df)
    return {
        'descritores': desc,
        'por_vagao': avaliar_vagoes(desc),
        'por_ponto': avaliar_pontos(desc),
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Análise de malha ferroviária por vagão e ponto.')
    parser.add_argument('csv_path', help='Caminho para o arquivo CSV de medidas de vibração')
    args = parser.parse_args()

    resultado = analisar_malha(args.csv_path)
    print('\nResumo por vagão:')
    print(resultado['por_vagao'].to_string(index=False))
    print('\nResumo por ponto:')
    print(resultado['por_ponto'].to_string(index=False))
