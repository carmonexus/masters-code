import pandas as pd
import numpy as np

# Parâmetros físicos do modelo SDOF (seção 12.2.2)
m = 32500
k = 1908100
zeta = 0.10
c = 2 * zeta * np.sqrt(k * m)
F0 = 22800
omega = 6

# Limites operacionais
LIMITE_VARIACAO_REL = 0.30   # 30 % de desvio relativo entre pontos
LIMITE_RMS_PONTO_G = 0.40    # RMS médio do ponto considerado alto


def amplitude_edo():
    """Amplitude teórica em regime permanente (seção 6.2)."""
    return F0 / np.sqrt((k - m * omega ** 2) ** 2 + (c * omega) ** 2)


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
    """Para cada vagão, mede a variação relativa do RMS entre os pontos da malha.
    Compara também a amplitude medida com a referência teórica da EDO."""
    A_ref_mm = 1000 * amplitude_edo()
    relatorio = []
    for vagao, g in desc.groupby('vagao_id'):
        rms = g['rms_g'].values
        if len(rms) < 2:
            continue
        media = float(np.mean(rms))
        variacao_rel = (float(np.max(rms)) - float(np.min(rms))) / media
        status = 'ALERTA' if variacao_rel > LIMITE_VARIACAO_REL else 'NORMAL'
        relatorio.append({
            'vagao_id': vagao,
            'pontos': int(len(rms)),
            'rms_min_g': float(np.min(rms)),
            'rms_max_g': float(np.max(rms)),
            'variacao_rel': variacao_rel,
            'A_edo_mm': A_ref_mm,
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
