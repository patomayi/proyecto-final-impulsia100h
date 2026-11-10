from pathlib import Path
import re
import unicodedata

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Esta ruta apunta al CSV aunque ejecutes la app desde otra carpeta.
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "datos" / "dataset_eco_asistente.csv"

# Traducimos las etiquetas del modelo a nombres de contenedores fáciles de entender.
NOMBRES_CONTENEDOR = {
    "papel y cartón": "Contenedor de papel y cartón (azul)",
    "vidrio": "Contenedor de vidrio (verde)",
    "envases": "Contenedor de plástico y metal (amarillo)",
    "orgánico": "Contenedor de biorresiduos (marrón)",
    "resto": "Contenedor de restos (gris)",
}

COLORES_CONTENEDOR = {
    "papel y cartón": "#4A90D9",
    "vidrio": "#32845F",
    "envases": "#E3B83E",
    "orgánico": "#8A6A4F",
    "resto": "#7D8884",
}

ESTILOS_RESULTADO = {
    "papel y cartón": ("#DCEBFA", "#275E95", "#B4D0EC"),
    "vidrio": ("#DDF0E5", "#286847", "#B8D9C3"),
    "envases": ("#FFF1C9", "#775A11", "#EBD58E"),
    "orgánico": ("#80512D", "#FFF8E9", "#654020"),
    "resto": ("#E6EBE9", "#4E5C55", "#CED8D3"),
}

CONSEJOS = {
    "papel y cartón": "Deposítalo limpio y, si es una caja, pliégala.",
    "vidrio": "Vacía el envase y separa tapas o tapones cuando corresponda.",
    "envases": "Vacía el envase antes de depositarlo.",
    "orgánico": "Deposita solo residuos orgánicos admitidos por el servicio local.",
    "resto": "Esta categoría es para residuos domésticos no incluidos en las fracciones selectivas.",
}

UMBRAL_CONFIDENCIA = 0.35
MENSAJE_PUNTO_LIMPIO = "Ir al punto limpio"
MENSAJE_MATERIAL_INDEFINIDO = "Necesito saber el material del envase"

# Listas sencillas para reconocer frutas y vegetales que el modelo de texto
# puede confundir aunque sean ejemplos comunes.
FRUTAS = {
    "manzana", "pera", "platano", "banana", "naranja", "mandarina",
    "limon", "lima", "uva", "fresa", "frutilla", "cereza", "melocoton",
    "durazno", "albaricoque", "damasco", "ciruela", "kiwi", "pina",
    "anana", "mango", "papaya", "melon", "sandia", "aguacate", "palta",
    "granada", "higo", "pomelo", "arandano", "frambuesa", "mora",
    "fruta", "frutas",
}

VEGETALES = {
    "lechuga", "espinaca", "acelga", "col", "repollo", "brocoli",
    "coliflor", "zanahoria", "patata", "papa", "batata", "boniato",
    "camote", "tomate", "pepino", "calabacin", "zucchini", "berenjena",
    "pimiento", "calabaza", "cebolla", "ajo", "puerro", "apio",
    "esparrago", "alcachofa", "judia verde", "judias verdes", "ejote",
    "guisante", "maiz", "maices", "elote", "yuca", "rabano", "remolacha",
    "nabo", "vegetal", "vegetales", "verdura", "verduras",
}

PALABRAS_ENVASE = {
    "botella", "envase", "frasco", "tarro", "lata", "brik", "bote",
    "bandeja", "bolsa", "envoltorio", "plastico", "vidrio", "cristal",
    "metal", "paquete",
}


def normalizar_texto(texto):
    return "".join(
        caracter for caracter in unicodedata.normalize("NFD", str(texto).lower())
        if unicodedata.category(caracter) != "Mn"
    )


def contiene_termino(texto, termino):
    # Acepta el singular y plurales sencillos, con límites de palabra para
    # evitar coincidencias accidentales dentro de otras palabras.
    termino = normalizar_texto(termino)
    if termino.endswith("z"):
        plural = termino[:-1] + "ces"
    elif termino.endswith(("a", "e", "i", "o", "u")):
        plural = termino + "s"
    else:
        plural = termino + "es"
    return any(
        re.search(r"(?<!\w)" + re.escape(variante) + r"(?!\w)", texto)
        for variante in {termino, plural}
    )


def detectar_tipo_fruta_vegetal(residuo):
    texto = normalizar_texto(residuo)
    # Si la descripción habla del envase o del material, se clasifica ese objeto
    # y no el alimento que quizá contenía.
    if any(contiene_termino(texto, palabra) for palabra in PALABRAS_ENVASE):
        return None
    if any(contiene_termino(texto, fruta) for fruta in FRUTAS):
        return "fruta"
    if any(contiene_termino(texto, vegetal) for vegetal in VEGETALES):
        return "vegetal"
    return None
PALABRAS_CLAVE_CATEGORIA = {
    "papel y cartón": [
        "papel", "carton", "cuaderno", "libro", "hoja", "revista", "folleto",
        "sobre", "catalogo", "postal", "caja", "bolsa de papel", "envoltorio de papel",
        "cartulina", "periódico", "diario", "agenda", "cuaderno", "libreta"
    ],
    "vidrio": [
        "vidrio", "frasco", "botella", "tarro", "jarra", "vaso de vidrio",
        "botellin", "botellita", "envase de vidrio", "cristal"
    ],
    "envases": [
        "envase", "plastico", "lata", "aluminio", "brik", "bote", "film", "tapón",
        "aerosol", "bolsa de plastico", "poliestireno", "metal", "tapa", "garrafa"
    ],
    "orgánico": [
        "cascara", "piel", "restos", "posos", "verdura", "fruta", "manzana",
        "naranja", "cafe", "pan", "arroz", "pasta", "cocinado", "queso", "yogur",
        "hojas", "pera", "banana", "ensalada", "brocoli", "tomate", "cebolla", "zanahoria"
    ],
    "resto": [
        "mascarilla", "colilla", "cepillo", "polvo", "bastoncillo", "toallita",
        "esponja", "ceramica", "porcelana", "juguete", "goma", "ropa", "zapato",
        "muñeco", "espejo", "bombilla", "jeringuilla", "dental", "uso", "sucia"
    ],
}


@st.cache_resource
# La caché evita volver a entrenar los modelos en cada clic o actualización de la página.
def cargar_recursos():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No encuentro el dataset: {DATASET_PATH}")

    # pandas lee el CSV y permite limpiar/revisar sus columnas con facilidad.
    datos = pd.read_csv(DATASET_PATH)
    datos = datos.dropna(subset=["residuo", "categoria"]).copy()
    datos["residuo"] = datos["residuo"].astype(str).str.strip().str.lower()
    datos["categoria"] = datos["categoria"].astype(str).str.strip().str.lower()
    datos = datos[(datos["residuo"] != "") & (datos["categoria"] != "")]

    textos = datos["residuo"]
    etiquetas = datos["categoria"]

    # Reservamos parte de los ejemplos; el modelo no los verá mientras aprende.
    textos_entrenamiento, textos_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(
        textos,
        etiquetas,
        test_size=0.25,
        random_state=42,
        stratify=etiquetas,
    )

    # CountVectorizer cuenta palabras y LogisticRegression aprende su relación con cada etiqueta.
    modelo_evaluacion = make_pipeline(
        CountVectorizer(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    modelo_evaluacion.fit(textos_entrenamiento, etiquetas_entrenamiento)
    predicciones = modelo_evaluacion.predict(textos_prueba)

    # Comparamos predicciones con etiquetas reales para medir el desempeño en datos reservados.
    exactitud = accuracy_score(etiquetas_prueba, predicciones)
    matriz = confusion_matrix(
        etiquetas_prueba,
        predicciones,
        labels=sorted(etiquetas.unique()),
    )
    resultados = pd.DataFrame({
        "Categoría": etiquetas_prueba.to_numpy(),
        "Resultado": [
            "Acierto" if real == predicha else "Error"
            for real, predicha in zip(etiquetas_prueba, predicciones)
        ],
    })

    # Para la app final entrenamos con todos los ejemplos disponibles.
    modelo_final = make_pipeline(
        CountVectorizer(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    modelo_final.fit(textos, etiquetas)

    return datos, modelo_final, exactitud, matriz, sorted(etiquetas.unique()), resultados


st.set_page_config(page_title="Eco-asistente", page_icon="♻️", layout="centered")
st.session_state.setdefault("sesion_cerrada", False)
st.session_state.setdefault("residuo_pendiente_material", None)
st.session_state.setdefault("material_seleccionado", "Elige un material")
st.session_state.setdefault("resultado_categoria", None)
st.session_state.setdefault("resultado_tipo", None)

st.markdown(
    """
    <style>
    [data-testid="stApp"] {
        --eco-forest: light-dark(#174A34, #B7DCC0);
        --eco-leaf: light-dark(#3D714C, #83BD91);
        --eco-sage: light-dark(#E8EBD9, #26342A);
        --eco-cream: light-dark(#F6F0DF, #141D17);
        --eco-page-top: light-dark(#F2EBDD, #101814);
        --eco-page-bottom: light-dark(#F6F0DF, #141D17);
        --eco-card: light-dark(#FFFCF4, #1B2820);
        --eco-card-border: light-dark(#D9C9A9, #405446);
        --eco-input: light-dark(#FFFEFA, #26332B);
        --eco-ink: light-dark(#2E3027, #F0F2E9);
        --eco-muted: light-dark(#6D7467, #AAB7AA);
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, var(--eco-page-top) 0%, var(--eco-page-bottom) 100%);
        color: var(--eco-ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 860px; padding-top: 2rem; padding-bottom: 2.5rem; }
    .eco-hero {
        padding: 1.7rem 1.9rem 2.35rem; margin-bottom: 1.2rem;
        border: 1px solid #315C43; border-radius: 24px;
        background: linear-gradient(145deg, #123F2D 0%, #20583D 68%, #2C6546 100%);
        box-shadow: 0 16px 34px rgba(39, 55, 35, .19);
        text-align: center;
        overflow: hidden;
        position: relative;
    }
    .eco-hero-plant { position: absolute; right: 5%; bottom: 1.1rem; width: 70px; height: 84px; opacity: .9; pointer-events: none; }
    .eco-hero h1 { color: #FFF9E9; font-family: Georgia, 'Times New Roman', serif; font-size: clamp(2.4rem, 6vw, 3.6rem); line-height: 1.05; margin: 0 0 1rem 0; font-weight: 800; }
    .eco-hero h2 { color: #FFF9E9; font-family: Georgia, 'Times New Roman', serif; font-size: clamp(1rem, 4.2vw, 2.4rem); line-height: 1.12; white-space: nowrap; margin: 0 0 .8rem 0; font-weight: 700; }
    .eco-hero p { color: #E4E9D6; font-size: 1rem; margin: 0; }
    .eco-hero::after {
        content: ""; position: absolute; left: -3%; bottom: -1px; width: 106%; height: 18px;
        background: var(--eco-page-bottom); border-radius: 55% 55% 0 0 / 100% 100% 0 0;
        box-shadow: 0 -3px 0 rgba(185, 139, 77, .8);
    }
    @media (max-width: 560px) { .eco-hero-plant { width: 42px; height: 58px; right: 3%; opacity: .65; } }
    [data-testid="stVerticalBlockBorderWrapper"] { border: 1px solid var(--eco-card-border) !important; border-radius: 20px !important; background: var(--eco-card) !important; box-shadow: 0 9px 24px rgba(30, 42, 32, .12); }
    .st-key-search-card, .st-key-explore-card,
    [data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-search-card),
    [data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-explore-card) {
        border: 1px solid var(--eco-card-border) !important;
        border-radius: 22px !important;
        background: var(--eco-card) !important;
        box-shadow: 0 12px 28px rgba(30, 42, 32, .14) !important;
    }
    .st-key-search-card, .st-key-explore-card {
        padding: .8rem 1.05rem !important;
    }
    [data-testid="stTextInputRootElement"],
    [data-testid="stTextInputRootElement"] > div,
    [data-testid="stTextInputRootElement"] [data-baseweb="input"],
    [data-testid="stTextInput"] [data-baseweb="input"],
    div[data-baseweb="input"], div[data-baseweb="base-input"] {
        background: var(--eco-input) !important;
        border-color: var(--eco-card-border) !important;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stTextInputRootElement"] input,
    input[type="text"] {
        min-height: 3rem; border-radius: 12px;
        background: var(--eco-input) !important; background-color: var(--eco-input) !important;
        color: var(--eco-ink) !important; -webkit-text-fill-color: var(--eco-ink) !important;
        caret-color: var(--eco-ink) !important; font-size: 1rem;
        opacity: 1 !important;
    }
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextInputRootElement"] input::placeholder,
    input[type="text"]::placeholder {
        color: var(--eco-muted) !important; -webkit-text-fill-color: var(--eco-muted) !important; opacity: 1;
    }
    [data-testid="stTextInput"] label,
    [data-testid="stTextInput"] label p,
    [data-testid="stWidgetLabel"] p {
        color: var(--eco-ink) !important; -webkit-text-fill-color: var(--eco-ink) !important;
    }
    [data-testid="stTextInputRootElement"] input:focus { border-color: var(--eco-leaf); box-shadow: 0 0 0 .15rem rgba(57,123,91,.14); }
    [data-testid="stBaseButton-primary"] {
        min-height: 3rem; border: 0 !important; border-radius: 12px !important;
        background: var(--eco-forest) !important; color: #fff !important; font-weight: 700;
    }
    [data-testid="stBaseButton-primary"] p { color: #fff !important; }
    [data-testid="stBaseButton-primary"]:hover { background: var(--eco-leaf) !important; }
    [data-testid="stBaseButton-secondary"] {
        min-height: 2.5rem; border-radius: 12px !important;
        border: 1px solid var(--eco-card-border) !important; background: var(--eco-sage) !important;
        color: var(--eco-forest) !important; font-weight: 700;
    }
    [data-testid="stBaseButton-secondary"] p { color: var(--eco-forest) !important; }
    [data-testid="stBaseButton-secondary"]:hover { background: var(--eco-cream) !important; border-color: var(--eco-leaf) !important; }
    .eco-result {
        position: relative; display: flex; flex-direction: row; align-items: center; gap: .8rem;
        width: min(100%, 350px); min-height: 96px; box-sizing: border-box;
        border: 1px solid; border-left-width: 5px; border-radius: 14px;
        padding: .8rem 2.6rem .8rem .8rem; margin: .35rem auto .8rem auto;
    }
    .eco-search-title { text-align: center; color: var(--eco-forest); margin-top: 0; }
    .eco-bin-art { display: block; flex: 0 0 54px; width: 54px; height: 64px; }
    .eco-plant-art { display: block; position: absolute; top: 5px; right: 7px; width: 30px; height: 34px; }
    .eco-result-copy { display: flex; flex-direction: column; gap: .25rem; min-width: 0; }
    .eco-result-label { font-size: .72rem; font-weight: 700; opacity: .9; }
    .eco-result-value { font-size: .94rem; font-weight: 800; line-height: 1.3; }
    @media (max-width: 560px) {
        .eco-result { gap: .6rem; padding: .7rem 2.3rem .7rem .7rem; min-height: 88px; }
        .eco-bin-art { flex-basis: 48px; width: 48px; height: 58px; }
        .eco-result-value { font-size: .9rem; }
    }
    [data-testid="stAlert"] { border-radius: 14px; }
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"],
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
        color: var(--eco-ink) !important;
    }
    [data-testid="stExpander"] { border: 1px solid var(--eco-card-border); border-radius: 16px; background: var(--eco-card); overflow: hidden; }
    [data-testid="stExpander"] details > summary {
        background: var(--eco-sage) !important; color: var(--eco-forest) !important;
        border-radius: 15px; font-weight: 700;
    }
    [data-testid="stExpander"] details > summary p,
    [data-testid="stExpander"] details > summary span,
    [data-testid="stExpander"] details > summary svg {
        color: var(--eco-forest) !important; fill: var(--eco-forest) !important;
    }
    [data-testid="stExpander"] details[open] > summary { border-bottom: 1px solid var(--eco-card-border); border-radius: 15px 15px 0 0; }
    h2, h3 { color: var(--eco-forest); font-family: Georgia, 'Times New Roman', serif; }
    .eco-footer { color: var(--eco-muted); font-size: .78rem; text-align: center; padding-top: .5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def cerrar_sesion():
    # Borra los datos de esta sesión del navegador; el servidor sigue activo en la terminal.
    for clave in list(st.session_state.keys()):
        del st.session_state[clave]
    st.session_state["sesion_cerrada"] = True


def detectar_categoria_o_punto_limpio(residuo, modelo=None, datos=None):
    # Si no se pasan los recursos, los cargamos juntos. La app pasa ambos
    # explícitamente para reutilizar los recursos ya cargados en su sesión.
    if modelo is None or datos is None:
        datos_cargados, modelo_cargado, *_ = cargar_recursos()
        datos = datos if datos is not None else datos_cargados
        modelo = modelo if modelo is not None else modelo_cargado

    texto = str(residuo or "").strip().lower()
    if not texto:
        return MENSAJE_PUNTO_LIMPIO

    coincidencias = datos[datos["residuo"].str.strip().str.lower() == texto]
    if not coincidencias.empty:
        return coincidencias.iloc[0]["categoria"]

    # Las reglas solo reconocen casos explícitos de residuos especiales que
    # deben salir de las cinco categorías del clasificador.
    texto_normalizado = "".join(
        caracter for caracter in unicodedata.normalize("NFD", texto)
        if unicodedata.category(caracter) != "Mn"
    )
    terminos_especiales = (
        "pila", "bateria", "medicamento", "jeringuilla", "aparato electrico",
        "electrodomestico", "ordenador", "portatil", "laptop", "movil",
        "telefono", "bombilla", "fluorescente", "punto limpio"
    )
    if any(termino in texto_normalizado for termino in terminos_especiales):
        return MENSAJE_PUNTO_LIMPIO

    palabras_envase = (
        "botella", "envase", "frasco", "tarro", "bote", "bandeja",
        "bolsa", "paquete"
    )
    menciona_envase = any(
        contiene_termino(texto_normalizado, palabra) for palabra in palabras_envase
    )
    menciona_vidrio = any(
        contiene_termino(texto_normalizado, palabra) for palabra in ("vidrio", "cristal")
    )
    menciona_ligero = any(
        contiene_termino(texto_normalizado, palabra)
        for palabra in ("plastico", "metal", "aluminio", "brik")
    )
    if menciona_vidrio and menciona_ligero:
        return MENSAJE_MATERIAL_INDEFINIDO
    if menciona_envase and menciona_vidrio:
        return "vidrio"
    if menciona_envase and menciona_ligero:
        return "envases"
    if menciona_envase:
        return MENSAJE_MATERIAL_INDEFINIDO

    # Las latas y los briks ya identifican el tipo de envase ligero aunque no
    # se indique el material en la frase.
    if contiene_termino(texto_normalizado, "lata") or contiene_termino(texto_normalizado, "brik"):
        return "envases"

    # El nombre de una bebida no indica por sí solo si se tira una botella,
    # una lata u otro tipo de envase.
    bebidas = ("cerveza", "vino", "refresco", "agua", "zumo", "leche")
    if any(contiene_termino(texto_normalizado, bebida) for bebida in bebidas):
        return MENSAJE_MATERIAL_INDEFINIDO

    if detectar_tipo_fruta_vegetal(texto):
        return "orgánico"

    # Las palabras clave solo determinan si la entrada pertenece a alguna de
    # las cinco fracciones conocidas; la categoría final siempre la decide el
    # clasificador, evitando que la primera regla coincidente fuerce una etiqueta.
    palabras_conocidas = {
        palabra
        for palabras in PALABRAS_CLAVE_CATEGORIA.values()
        for palabra in palabras
    }
    if not any(
        "".join(caracter for caracter in unicodedata.normalize("NFD", palabra)
                if unicodedata.category(caracter) != "Mn") in texto_normalizado
        for palabra in palabras_conocidas
    ):
        return MENSAJE_PUNTO_LIMPIO

    return modelo.predict([texto])[0]


SVG_CONTENEDOR_PLANTA = """<svg class="eco-bin-art" viewBox="0 0 64 72" role="img" aria-label="Contenedor de reciclaje">
<path d="M13 18h38l-3.5 46h-31L13 18Z" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linejoin="round"/>
<path d="M9 14h46M23 14l2-6h14l2 6M23 68v2m18-2v2" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>
<text x="17" y="49" font-size="25" fill="currentColor" font-family="Arial, sans-serif">♻</text>
</svg><svg class="eco-plant-art" viewBox="0 0 40 44" role="img" aria-label="Plantita">
<path d="M18 37c1-9 2-17 9-25" fill="none" stroke="#55783D" stroke-width="3" stroke-linecap="round"/>
<path d="M24 15C15 16 10 12 9 5c8-2 14 2 15 10Zm3-4C29 4 34 1 39 2c0 7-4 12-12 12Z" fill="#A8C76E" stroke="#55783D" stroke-width="2" stroke-linejoin="round"/>
</svg>"""


def mostrar_resultado(categoria, tipo_detectado=None):
    with st.container(border=True):
        st.markdown("#### Resultado")
        if tipo_detectado:
            st.info(f"Tipo identificado: **{tipo_detectado.capitalize()}**")
        if categoria == MENSAJE_PUNTO_LIMPIO:
            st.markdown(
                '<div class="eco-result" style="background:#FCE4E4;color:#973F3F;border-color:#EFBDBD;">'
                f'{SVG_CONTENEDOR_PLANTA}'
                '<div class="eco-result-copy"><span class="eco-result-label">Orientación</span>'
                '<span class="eco-result-value">Ir al punto limpio</span></div></div>',
                unsafe_allow_html=True,
            )
        else:
            fondo, texto_color, borde = ESTILOS_RESULTADO.get(
                categoria, ("#EEF2EF", "#33443B", "#D6E0D8")
            )
            destino = NOMBRES_CONTENEDOR.get(categoria, categoria)
            st.markdown(
                f'<div class="eco-result" style="background:{fondo};color:{texto_color};border-color:{borde};">'
                f'{SVG_CONTENEDOR_PLANTA}'
                f'<div class="eco-result-copy"><span class="eco-result-label">Deposítalo en</span>'
                f'<span class="eco-result-value">{destino}</span></div></div>',
                unsafe_allow_html=True,
            )
            st.write(CONSEJOS.get(categoria, "Consulta las instrucciones de separación locales."))


def explicar_decision(residuo, categoria, modelo, datos):
    texto = str(residuo or "").strip().lower()
    coincidencias = datos[datos["residuo"].str.strip().str.lower() == texto]
    if not coincidencias.empty:
        return (
            "Coincidencia exacta con el CSV",
            f'La descripción aparece en el CSV principal con la etiqueta "{coincidencias.iloc[0]["categoria"]}". Para esta consulta no hizo falta que el modelo generalizara.',
        )

    normalizado = normalizar_texto(texto)
    terminos_especiales = (
        "pila", "bateria", "medicamento", "jeringuilla", "aparato electrico",
        "electrodomestico", "ordenador", "portatil", "laptop", "movil",
        "telefono", "bombilla", "fluorescente", "punto limpio"
    )
    termino_especial = next((t for t in terminos_especiales if t in normalizado), None)
    if termino_especial:
        return "Regla para residuos especiales", f'La regla de la app reconoce "{termino_especial}" y recomienda revisar el punto limpio.'

    palabras_envase = ("botella", "envase", "frasco", "tarro", "bote", "bandeja", "bolsa", "paquete")
    menciona_envase = any(contiene_termino(normalizado, p) for p in palabras_envase)
    menciona_vidrio = any(contiene_termino(normalizado, p) for p in ("vidrio", "cristal"))
    menciona_ligero = any(contiene_termino(normalizado, p) for p in ("plastico", "metal", "aluminio", "brik"))
    if categoria == MENSAJE_MATERIAL_INDEFINIDO:
        return "Aclaración pendiente", "La descripción parece referirse a una bebida o a un envase, pero falta indicar qué se va a tirar o de qué material es. Todavía no se ha asignado una categoría."
    if menciona_envase and (menciona_vidrio or menciona_ligero):
        return "Regla del material del envase", "La app usa el material escrito: vidrio/cristal corresponde a vidrio; plástico, metal/aluminio o brik corresponde a envases."
    if contiene_termino(normalizado, "lata") or contiene_termino(normalizado, "brik"):
        return "Regla del tipo de envase", "La app reconoce una lata o un brik como envase ligero y lo asigna a envases."
    if detectar_tipo_fruta_vegetal(texto):
        tipo = detectar_tipo_fruta_vegetal(texto)
        return "Lista explícita de frutas y vegetales", f'La palabra coincide con la lista programada de "{tipo}" y esa regla la asigna a orgánico. Esta respuesta no procede del modelo entrenado.'

    palabras_conocidas = {
        palabra
        for palabras in PALABRAS_CLAVE_CATEGORIA.values()
        for palabra in palabras
    }
    tiene_pista = any(
        normalizar_texto(palabra) in normalizado for palabra in palabras_conocidas
    )
    if not tiene_pista:
        return "Sin coincidencia suficiente", "La app no encontró una frase exacta ni una palabra clave que active el modelo. El aviso de punto limpio es una salida de seguridad de la app, no una predicción aprendida para este producto."

    vectorizador = modelo.named_steps["countvectorizer"]
    clasificador = modelo.named_steps["logisticregression"]
    vector = vectorizador.transform([texto])
    indice_categoria = list(clasificador.classes_).index(categoria)
    nombres = vectorizador.get_feature_names_out()
    pesos = clasificador.coef_[indice_categoria]
    contribuciones = [
        (nombres[indice], float(vector[0, indice] * pesos[indice]))
        for indice in vector[0].indices
        if pesos[indice] > 0
    ]
    contribuciones.sort(key=lambda item: item[1], reverse=True)
    terminos = [palabra for palabra, _ in contribuciones[:4]]
    if terminos:
        evidencia = ", ".join(f'"{palabra}"' for palabra in terminos)
        explicacion = f'El modelo usó palabras presentes en la descripción que tienen peso positivo para "{categoria}": {evidencia}. Aprendió esas asociaciones de los ejemplos del CSV.'
    else:
        explicacion = f'El modelo comparó las palabras de la descripción con los patrones aprendidos y eligió "{categoria}". No encontró una palabra individual con peso positivo claro para mostrar.'
    return "Modelo de texto", explicacion


if st.session_state["sesion_cerrada"]:
    st.title("Sesión cerrada")
    st.info("Tus datos de esta sesión se han borrado. Para apagar la aplicación local, pulsa Ctrl+C en la terminal donde ejecutaste Streamlit.")
    st.stop()

st.markdown(
    """
    <div class="eco-hero">
      <svg class="eco-hero-plant" viewBox="0 0 64 80" aria-hidden="true">
        <path d="M29 74c2-19 7-35 20-51" fill="none" stroke="#B9D28A" stroke-width="4" stroke-linecap="round"/>
        <path d="M43 34C25 37 15 29 14 14c16-3 27 5 29 19Zm7-13C51 6 61 1 63 2c1 15-4 24-14 27Z" fill="#A9C86C" stroke="#D7E6B2" stroke-width="2" stroke-linejoin="round"/>
        <path d="M34 60C20 62 10 55 8 43c14-2 24 4 27 16Z" fill="#789C53" stroke="#B9D28A" stroke-width="2" stroke-linejoin="round"/>
      </svg>
      <h1>♻ Eco-asistente</h1>
      <h2>Reciclar empieza por identificarlo</h2>
      <p>Escribe un residuo y recibe una orientación clara sobre dónde depositarlo.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    datos, modelo, exactitud, matriz, categorias, resultados = cargar_recursos()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

with st.container(border=True, key="search-card"):
    st.markdown('<h3 class="eco-search-title">¿Dónde lo tiramos?</h3>', unsafe_allow_html=True)
    st.caption("Prueba con un nombre y material: botella de plástico, frasco de vidrio o lechuga.")
    residuo = st.text_input(
        "Nombre del residuo",
        placeholder="Ej.: botella de tomate de vidrio",
        label_visibility="collapsed",
    )
    _, boton_col, _ = st.columns([1, 1, 1])
    with boton_col:
        consultar = st.button("♻ Clasificar", type="primary", use_container_width=True)

    if consultar:
        if not residuo.strip():
            st.session_state["residuo_pendiente_material"] = None
            st.session_state["resultado_categoria"] = None
            st.session_state["resultado_tipo"] = None
            st.warning("Escribe un residuo para obtener una orientación.")
        else:
            tipo_detectado = detectar_tipo_fruta_vegetal(residuo)
            categoria = detectar_categoria_o_punto_limpio(residuo, modelo, datos)
            if categoria == MENSAJE_MATERIAL_INDEFINIDO:
                st.session_state["residuo_pendiente_material"] = residuo.strip()
                st.session_state["material_seleccionado"] = "Elige un material"
                st.session_state["resultado_categoria"] = None
                st.session_state["resultado_tipo"] = None
            else:
                st.session_state["residuo_pendiente_material"] = None
                st.session_state["resultado_categoria"] = categoria
                st.session_state["resultado_tipo"] = tipo_detectado

    if st.session_state["residuo_pendiente_material"]:
        st.markdown("#### ¿De qué material es el envase?")
        st.caption(f"Residuo: {st.session_state['residuo_pendiente_material']}")
        opciones_material = [
            "Elige un material",
            "Vidrio o cristal",
            "Plástico",
            "Metal o aluminio",
            "Brik",
        ]
        st.selectbox(
            "Material",
            opciones_material,
            key="material_seleccionado",
            label_visibility="collapsed",
        )
        if st.button("Confirmar material", key="confirmar_material", type="primary"):
            categoria_por_material = {
                "Vidrio o cristal": "vidrio",
                "Plástico": "envases",
                "Metal o aluminio": "envases",
                "Brik": "envases",
            }
            categoria_elegida = categoria_por_material.get(st.session_state["material_seleccionado"])
            if categoria_elegida is None:
                st.warning("Elige un material antes de continuar.")
            else:
                st.session_state["resultado_categoria"] = categoria_elegida
                st.session_state["resultado_tipo"] = None
                st.session_state["residuo_pendiente_material"] = None
                st.rerun()

    if st.session_state["resultado_categoria"] is not None:
        mostrar_resultado(
            st.session_state["resultado_categoria"],
            st.session_state["resultado_tipo"],
        )

# Estas gráficas ayudan a explicar tanto los datos como los aciertos y errores del modelo.
with st.expander("📊 Ver análisis de datos y evaluación del modelo"):
    st.caption("Estos gráficos muestran una evaluación fija sobre ejemplos reservados. No cambian al consultar un residuo nuevo.")
    st.subheader("Ejemplos disponibles por categoría")
    conteo_categorias = (
        datos["categoria"]
        .value_counts()
        .reindex(categorias, fill_value=0)
        .rename_axis("Categoría")
        .reset_index(name="Número de ejemplos")
    )
    grafico_conteo = alt.Chart(conteo_categorias).mark_bar(
        cornerRadiusTopLeft=6, cornerRadiusTopRight=6
    ).encode(
        x=alt.X("Categoría:N", sort=categorias, axis=alt.Axis(title=None, labelAngle=-18)),
        y=alt.Y("Número de ejemplos:Q", axis=alt.Axis(title="Ejemplos", gridColor="#E7ECE7")),
        color=alt.Color(
            "Categoría:N",
            scale=alt.Scale(domain=categorias, range=[COLORES_CONTENEDOR[c] for c in categorias]),
            legend=None,
        ),
        tooltip=["Categoría:N", "Número de ejemplos:Q"],
    ).properties(height=250)
    grafico_conteo = grafico_conteo.configure(background="#FFFFFF").configure_view(fill="#FFFFFF", strokeOpacity=0).configure_axis(
        labelColor="#52645A", titleColor="#52645A", tickColor="#DCE5DC"
    )
    st.altair_chart(grafico_conteo, use_container_width=True)

    st.subheader("Aciertos y errores en los residuos de prueba")
    st.write(f"Exactitud en el conjunto de prueba: **{exactitud:.0%}**")
    resumen = (
        resultados.groupby(["Categoría", "Resultado"])
        .size()
        .unstack(fill_value=0)
        .reindex(categorias, fill_value=0)
        .reset_index()
        .melt(id_vars="Categoría", var_name="Resultado", value_name="Número de residuos")
    )
    grafico_resultados = alt.Chart(resumen).mark_bar(
        cornerRadiusTopLeft=4, cornerRadiusTopRight=4
    ).encode(
        x=alt.X("Categoría:N", sort=categorias, axis=alt.Axis(title=None, labelAngle=-18)),
        xOffset=alt.XOffset("Resultado:N", sort=["Acierto", "Error"]),
        y=alt.Y("Número de residuos:Q", axis=alt.Axis(title="Residuos", gridColor="#E7ECE7")),
        color=alt.Color(
            "Categoría:N",
            scale=alt.Scale(domain=categorias, range=[COLORES_CONTENEDOR[c] for c in categorias]),
            legend=None,
        ),
        opacity=alt.Opacity(
            "Resultado:N",
            scale=alt.Scale(domain=["Acierto", "Error"], range=[1, 0.38]),
            legend=alt.Legend(title="Resultado", orient="top"),
        ),
        tooltip=["Categoría:N", "Resultado:N", "Número de residuos:Q"],
    ).properties(height=270)
    grafico_resultados = grafico_resultados.configure(background="#FFFFFF").configure_view(fill="#FFFFFF", strokeOpacity=0).configure_axis(
        labelColor="#52645A", titleColor="#52645A", tickColor="#DCE5DC"
    )
    st.altair_chart(grafico_resultados, use_container_width=True)

    st.subheader("Matriz de confusión")
    st.caption("Las filas son las categorías reales; las columnas, las predichas. La diagonal contiene los aciertos.")
    celdas_matriz = []
    for fila, categoria_real in enumerate(categorias):
        for columna, categoria_predicha in enumerate(categorias):
            diagonal = categoria_real == categoria_predicha
            color_fondo = COLORES_CONTENEDOR[categoria_real] if diagonal else "#F5F8F4"
            color_texto = "#3F3215" if diagonal and categoria_real == "envases" else ("#FFFFFF" if diagonal else "#26352D")
            celdas_matriz.append({
                "Categoría real": categoria_real,
                "Predicción": categoria_predicha,
                "Ejemplos": int(matriz[fila, columna]),
                "Color": color_fondo,
                "Color texto": color_texto,
            })
    datos_matriz = pd.DataFrame(celdas_matriz)
    celdas = alt.Chart(datos_matriz).mark_rect(stroke="#D9E3DB", strokeWidth=1).encode(
        x=alt.X("Predicción:N", sort=categorias, axis=alt.Axis(title=None, labelAngle=-18, ticks=False, domain=False)),
        y=alt.Y("Categoría real:N", sort=list(reversed(categorias)), axis=alt.Axis(title=None, ticks=False, domain=False)),
        color=alt.Color("Color:N", scale=None, legend=None),
        tooltip=["Categoría real:N", "Predicción:N", "Ejemplos:Q"],
    )
    numeros = alt.Chart(datos_matriz).mark_text(fontSize=13, fontWeight="bold").encode(
        x=alt.X("Predicción:N", sort=categorias),
        y=alt.Y("Categoría real:N", sort=list(reversed(categorias))),
        text=alt.Text("Ejemplos:Q", format="d"),
        color=alt.Color("Color texto:N", scale=None, legend=None),
    )
    grafico_matriz = (celdas + numeros).properties(height=260).configure(
        background="#FFFFFF"
    ).configure_view(fill="#FFFFFF", strokeOpacity=0).configure_axis(
        labelColor="#52645A", labelFontSize=11, titleColor="#52645A"
    )
    st.altair_chart(grafico_matriz, use_container_width=True)
    st.caption("La evaluación es inicial: el dataset es pequeño y las normas pueden variar según el municipio.")

with st.container(border=True, key="explore-card"):
    st.subheader("Explora la decisión")
    st.caption("Escribe un producto para ver la categoría y qué camino usó la app: CSV, regla o modelo.")
    producto_explicacion = st.text_input(
        "Producto para explicar",
        placeholder="Ej.: pan de molde viejo",
        key="producto_explicacion",
    )
    if st.button("¿Cómo lo clasificó?", key="explicar_producto", use_container_width=True):
        if not producto_explicacion.strip():
            st.warning("Escribe un producto para analizar la decisión.")
        else:
            categoria_explicacion = detectar_categoria_o_punto_limpio(
                producto_explicacion, modelo, datos
            )
            ruta, detalle = explicar_decision(
                producto_explicacion, categoria_explicacion, modelo, datos
            )
            st.markdown(f"**Ruta usada:** {ruta}")
            if categoria_explicacion == MENSAJE_MATERIAL_INDEFINIDO:
                st.info(detalle)
            else:
                mostrar_resultado(
                    categoria_explicacion,
                    detectar_tipo_fruta_vegetal(producto_explicacion),
                )
                st.write(f"**Qué influyó:** {detalle}")

st.markdown('<div class="eco-footer">Proyecto educativo · Orientación de reciclaje</div>', unsafe_allow_html=True)
st.markdown('<div class="eco-footer">®2026 - Emma, Maria y Priscila</div>', unsafe_allow_html=True)

st.divider()
st.button("Salir", on_click=cerrar_sesion, help="Cierra esta sesión y borra sus datos.")
st.caption(
    "La predicción es orientativa: el modelo se entrenó con un dataset educativo pequeño. "
    "En España, el sistema de recogida y algunas instrucciones pueden variar según el municipio. "
    "Para residuos especiales o voluminosos, consulta el servicio local o un punto limpio."
)