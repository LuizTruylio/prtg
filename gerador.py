'''
Mini projeto 1 - Gerador de números aleatórios com circuito quântico

Cenário 1: distribuição uniforme       (H em todos os qubits)
Cenário 2: distribuição tendenciosa    (H -> Rz(phi) -> H em todos os qubits)

Dependências: pip install qiskit qiskit-aer matplotlib numpy
'''
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ----------------------------------------------------------------------
# Parâmetros (altere aqui)
# ----------------------------------------------------------------------
N_BITS = 10                # 2**10 = 1024 valores possíveis (0..1023)
LIMITE = 1000              # maior número aceito (valores acima são descartados)
PHI = 2 * np.pi / 3        # ângulo da Rz -> define a tendência do cenário 2
QUANTIDADE = 5000          # quantos números gerar em cada cenário
DESENHAR_CIRCUITOS = True  # mostrar o desenho dos circuitos

backend = AerSimulator()


# ----------------------------------------------------------------------
# Circuitos
# ----------------------------------------------------------------------
def circuito_uniforme(n=N_BITS):
    '''H em cada qubit: cada bit vira uma "moeda justa" (50% / 50%).'''
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


def circuito_tendencioso(n=N_BITS, phi=PHI):
    '''
    H -> Rz(phi) -> H em cada qubit.
    A Rz só muda a fase; o segundo H converte essa fase em probabilidade:
    P(medir 1) = sin^2(phi/2). Com phi = 2*pi/3, P(1) = 75%.
    '''
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    qc.rz(phi, range(n))
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


# ----------------------------------------------------------------------
# Geração dos números
# ----------------------------------------------------------------------
def gerar_numeros(qc, quantidade, limite=LIMITE):
    '''
    Executa o circuito várias vezes (shots). Cada execução devolve uma
    cadeia de bits, convertida em inteiro. Se o valor passar do limite,
    é descartado e sorteado de novo (não usamos %, que distorceria a
    distribuição).
    '''
    qc_compilado = transpile(qc, backend)
    numeros = []
    while len(numeros) < quantidade:
        faltam = quantidade - len(numeros)
        resultado = backend.run(qc_compilado, shots=faltam, memory=True).result()
        for bits in resultado.get_memory():   # uma cadeia de bits por shot
            valor = int(bits, 2)              # binário -> inteiro
            if valor <= limite:
                numeros.append(valor)
    return np.array(numeros[:quantidade])


def frequencia_de_uns(numeros, n=N_BITS):
    '''Fração de bits 1 em cada posição (bit 0 = menos significativo).'''
    bits = (numeros[:, None] >> np.arange(n)) & 1
    return bits.mean(axis=0)


# ----------------------------------------------------------------------
# Execução e análise
# ----------------------------------------------------------------------
if __name__ == "__main__":
    qc_uniforme = circuito_uniforme()
    qc_tendencioso = circuito_tendencioso()

    if DESENHAR_CIRCUITOS:
        qc_uniforme.draw('mpl', fold=-1)
        qc_tendencioso.draw('mpl', fold=-1)

    uniformes = gerar_numeros(qc_uniforme, QUANTIDADE)
    tendenciosos = gerar_numeros(qc_tendencioso, QUANTIDADE)

    p_teorico = np.sin(PHI / 2) ** 2

    print(f"=== Cenário 1: uniforme ({QUANTIDADE} números) ===")
    print(f"Primeiros 10 números: {uniformes[:10].tolist()}")
    print(f"Média: {uniformes.mean():.1f} (esperado ~ {LIMITE / 2:.0f})")
    print(f"P(bit = 1) medida por posição: {np.round(frequencia_de_uns(uniformes), 3)}")
    print("P(bit = 1) teórica: 0.500\n")

    print(f"=== Cenário 2: tendencioso, phi = {PHI:.4f} rad ({QUANTIDADE} números) ===")
    print(f"Primeiros 10 números: {tendenciosos[:10].tolist()}")
    print(f"Média: {tendenciosos.mean():.1f}")
    print(f"P(bit = 1) medida por posição: {np.round(frequencia_de_uns(tendenciosos), 3)}")
    print(f"P(bit = 1) teórica: {p_teorico:.3f}")

    # Histogramas
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for eixo, dados, titulo in [(eixos[0], uniformes, "Cenário 1: uniforme"),
                                (eixos[1], tendenciosos, "Cenário 2: tendencioso")]:
        eixo.hist(dados, bins=25, range=(0, LIMITE), edgecolor="black")
        eixo.axvline(dados.mean(), color="red", linestyle="--",
                     label=f"média = {dados.mean():.0f}")
        eixo.set_title(titulo)
        eixo.set_xlabel("Número gerado")
        eixo.legend()
    eixos[0].set_ylabel("Frequência")
    plt.tight_layout()
    plt.show()