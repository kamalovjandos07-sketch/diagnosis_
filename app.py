import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Настройки страницы
st.set_page_config(
    page_title="Medical Diagnostic System",
    page_icon="🩺", 
    layout="wide"
)

# Стили
st.markdown("""
<style>
    .main { background-color: #f0f8ff; }
    .diagnosis-box { 
        background: white; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #228b22;
        margin: 10px 0;
    }
    .treatment-box {
        background: #e8f5e8;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .warning-box {
        background: #ffeaa7;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #fdcb6e;
    }
</style>
""", unsafe_allow_html=True)

# БАЗА ЗАБОЛЕВАНИЙ И ЛЕЧЕНИЯ
MEDICAL_KNOWLEDGE_BASE = {
    "community_acquired_pneumonia": {
        "diagnosis_criteria": ["Лихорадка >38°C", "Кашель", "Одышка", "Боль в груди", "Лейкоцитоз", "Повышение СРБ"],
        "required_criteria": 3,
        "treatments": {
            "antibiotics": ["Амоксициллин/клавуланат 875/125 мг 2 раза/сут × 7-10 дней", "Азитромицин 500 мг/сут × 3-5 дней"],
            "symptomatic": ["Парацетамол 500 мг при температуре", "Муколитики (АЦЦ 600 мг/сут)", "Ингаляции с физраствором"],
            "supportive": ["Постельный режим", "Обильное питье", "Контроль сатурации"]
        },
        "referral": "При тяжелом течении - госпитализация",
        "source": "IDSA/ATS Guidelines 2019"
    },
    
    "streptococcal_pharyngitis": {
        "diagnosis_criteria": ["Боль в горле", "Лихорадка >38°C", "Налеты на миндалинах", "Увеличение шейных лимфоузлов", "Отсутствие кашля"],
        "required_criteria": 4,
        "treatments": {
            "antibiotics": ["Феноксиметилпенициллин 500 мг 3 раза/сут × 10 дней", "Азитромицин 500 мг/сут × 3 дня при аллергии"],
            "symptomatic": ["Парацетамол 500 мг при боли", "Местные антисептики (Гексорал, Тантум Верде)", "Полоскание содо-солевым раствором"],
            "supportive": ["Щадящая диета", "Теплое питье", "Голосовой покой"]
        },
        "referral": "При рецидивирующем течении - консультация ЛОРа",
        "source": "IDSA Pharyngitis Guidelines"
    },
    
    "urinary_tract_infection": {
        "diagnosis_criteria": ["Дизурия", "Учащенное мочеиспускание", "Боль в надлобковой области", "Лихорадка", "Лейкоциты в моче"],
        "required_criteria": 2,
        "treatments": {
            "antibiotics": ["Нитрофурантоин 100 мг 3 раза/сут × 5 дней", "Фосфомицин 3 г однократно", "Цефтриаксон 1 г/сут в/м при осложнениях"],
            "symptomatic": ["Ибупрофен 400 мг при боли", "Спазмолитики (Но-шпа 40-80 мг/сут)", "Уросептики (Фитолизин)"],
            "supportive": ["Обильное питье", "Клюквенные морсы", "Исключение острой пищи"]
        },
        "referral": "При рецидивах - уролог, при беременности - срочно к врачу",
        "source": "IDSA UTI Guidelines"
    },
    
    "acute_bronchitis": {
        "diagnosis_criteria": ["Кашель <3 недель", "Может быть продуктивным", "Отсутствие лихорадки >38°C", "Отсутствие одышки", "Нормальные показатели воспаления"],
        "required_criteria": 3,
        "treatments": {
            "antibiotics": ["Антибиотики НЕ ПОКАЗАНЫ при вирусной этиологии"],
            "symptomatic": ["Противокашлевые (Синекод) при сухом кашле", "Муколитики (Амброксол 30 мг 3 раза/сут)", "Бронходилататоры (Сальбутамол) при бронхоспазме"],
            "supportive": ["Увлажнение воздуха", "Теплое питье", "Ингаляции", "Отказ от курения"]
        },
        "referral": "При сохранении симптомов >3 недель - пульмонолог",
        "source": "NICE Bronchitis Guidelines"
    },
    
    "influenza": {
        "diagnosis_criteria": ["Внезапное начало", "Лихорадка", "Головная боль", "Мышечные боли", "Слабость", "Сезонность"],
        "required_criteria": 3,
        "treatments": {
            "antivirals": ["Осельтамивир 75 мг 2 раза/сут × 5 дней", "Занамивир ингаляционно"],
            "symptomatic": ["Парацетамол 500 мг при температуре", "Ибупрофен 400 мг при боли", "Сосудосуживающие капли при рините"],
            "supportive": ["Постельный режим", "Обильное питье", "Витамин C", "Проветривание помещения"]
        },
        "referral": "При тяжелом течении, беременным, пожилым - срочно к врачу",
        "source": "WHO Influenza Guidelines"
    },
    
    "acute_gastroenteritis": {
        "diagnosis_criteria": ["Тошнота", "Рвота", "Диарея", "Боль в животе", "Слабость", "Возможна субфебрильная температура"],
        "required_criteria": 3,
        "treatments": {
            "rehydration": ["Регидрон 1 пакет на 1 л воды", "Оральные солевые растворы", "Частое дробное питье"],
            "symptomatic": ["Смекта 3 пакета/сут", "Энтеросорбенты (Полисорб)", "Противорвотные (Метоклопрамид) только по назначению"],
            "diet": ["Голод 4-6 часов", "Затем щадящая диета (рис, сухари, бананы)", "Исключение молочного, жирного, острого"]
        },
        "referral": "При признаках дегидратации, крови в стуле - срочно к врачу",
        "source": "ESPID Gastroenteritis Guidelines"
    },
    
    "hypertensive_crisis": {
        "diagnosis_criteria": ["АД >180/120 мм рт.ст.", "Головная боль", "Тошнота", "Нарушение зрения", "Одышка", "Боль в груди"],
        "required_criteria": 2,
        "treatments": {
            "emergency": ["Немедленный вызов скорой помощи", "Каптоприл 25 мг сублингвально", "Нифедипин 10 мг (только по назначению)"],
            "monitoring": ["Контроль АД каждые 15 минут", "Покой, полусидячее положение", "Доступ свежего воздуха"]
        },
        "referral": "ЭКГ, госпитализация в кардиологическое отделение",
        "source": "ESC Hypertension Guidelines"
    },
    
    "migraine": {
        "diagnosis_criteria": ["Пульсирующая головная боль", "Односторонняя локализация", "Тошнота/рвота", "Фоно/фотофобия", "Аура"],
        "required_criteria": 3,
        "treatments": {
            "acute": ["Суматриптан 50-100 мг", "Ибупрофен 400-600 мг", "Парацетамол 500-1000 мг"],
            "symptomatic": ["Противорвотные (Метоклопрамид 10 мг)", "Покой в темной комнате", "Холод на лоб"],
            "prophylaxis": ["Пропранолол 40-80 мг/сут", "Топирамат 25-50 мг/сут", "Исключение триггеров"]
        },
        "referral": "При частых приступах - невролог",
        "source": "IHS Migraine Guidelines"
    },
    
    "allergic_rhinitis": {
        "diagnosis_criteria": ["Чихание", "Ринорея", "Заложенность носа", "Зуд в носу", "Слезотечение", "Сезонность"],
        "required_criteria": 3,
        "treatments": {
            "antihistamines": ["Лоратадин 10 мг/сут", "Цетиризин 10 мг/сут", "Фексофенадин 180 мг/сут"],
            "nasal": ["Интраназальные кортикостероиды (Мометазон)", "Азеластин назальный спрей", "Солевые растворы"],
            "avoidance": ["Исключение аллергенов", "Влажная уборка", "Воздушные фильтры"]
        },
        "referral": "При неэффективности терапии - аллерголог",
        "source": "ARIA Guidelines"
    }
}

# ДИАГНОСТИЧЕСКАЯ СИСТЕМА
def medical_diagnosis_system(symptoms, lab_data, vital_signs, temperature, bp_systolic, bp_diastolic, wbc, crp):
    """
    Умная диагностическая система на основе баллов
    """
    symptom_score = {}
    
    # Проверяем критические состояния первыми
    if bp_systolic > 180 and bp_diastolic > 120:
        if any(symptom in ["Головная боль", "Тошнота", "Нарушение зрения", "Одышка", "Боль в груди"] for symptom in symptoms):
            return "hypertensive_crisis", 10
    
    # Определяем лабораторные показатели
    has_leukocytosis = "Лейкоцитоз" in lab_data or wbc > 10.0
    has_elevated_crp = "Повышение СРБ" in lab_data or crp > 5.0
    has_urinary_leuko = "Лейкоциты в моче" in lab_data
    
    # Пневмония
    pneumonia_score = sum([
        2 if "Лихорадка >38°C" in symptoms and temperature > 38 else 0,
        2 if "Кашель с мокротой" in symptoms else 1 if "Кашель" in symptoms else 0,
        2 if "Одышка" in symptoms else 0,
        2 if "Боль в груди" in symptoms else 0,
        2 if has_leukocytosis else 0,
        2 if has_elevated_crp else 0
    ])
    symptom_score["community_acquired_pneumonia"] = pneumonia_score
    
    # Ангина
    pharyngitis_score = sum([
        2 if "Боль в горле" in symptoms else 0,
        2 if "Налеты на миндалинах" in symptoms else 0,
        2 if "Лихорадка >38°C" in symptoms and temperature > 38 else 0,
        2 if "Увеличение лимфоузлов" in symptoms else 0,
        -2 if "Кашель" in symptoms else 1,
        1 if "Головная боль" in symptoms else 0
    ])
    symptom_score["streptococcal_pharyngitis"] = pharyngitis_score
    
    # ИМП
    uti_score = sum([
        3 if "Дизурия" in symptoms else 0,
        2 if "Учащенное мочеиспускание" in symptoms else 0,
        2 if "Боль в надлобковой области" in symptoms else 0,
        2 if has_urinary_leuko else 0,
        2 if "Лихорадка >38°C" in symptoms and temperature > 38 else 0
    ])
    symptom_score["urinary_tract_infection"] = uti_score
    
    # Бронхит
    bronchitis_score = sum([
        2 if "Кашель" in symptoms else 0,
        2 if "Кашель с мокротой" in symptoms else 0,
        -2 if "Лихорадка >38°C" in symptoms and temperature > 38 else 1,
        -2 if "Одышка" in symptoms else 1,
        -2 if has_leukocytosis else 1,
        1 if "Слабость" in symptoms else 0
    ])
    symptom_score["acute_bronchitis"] = bronchitis_score
    
    # Грипп
    influenza_score = sum([
        2 if "Лихорадка >38°C" in symptoms and temperature > 38 else 0,
        2 if "Головная боль" in symptoms else 0,
        2 if "Мышечные боли" in symptoms else 0,
        2 if "Слабость" in symptoms else 0,
        2 if "Внезапное начало" in symptoms else 0,
        1 if "Сезонность" in symptoms else 0
    ])
    symptom_score["influenza"] = influenza_score
    
    # Гастроэнтерит
    gastroenteritis_score = sum([
        3 if "Тошнота" in symptoms else 0,
        3 if "Рвота" in symptoms else 0,
        3 if "Диарея" in symptoms else 0,
        2 if "Боль в животе" in symptoms else 0,
        1 if "Слабость" in symptoms else 0,
        1 if "Субфебрильная температура" in symptoms and 37 < temperature < 38 else 0
    ])
    symptom_score["acute_gastroenteritis"] = gastroenteritis_score
    
    # Мигрень
    migraine_score = sum([
        3 if "Пульсирующая головная боль" in symptoms else 0,
        2 if "Односторонняя локализация" in symptoms else 0,
        2 if "Тошнота/рвота" in symptoms else 0,
        2 if "Фоно/фотофобия" in symptoms else 0,
        3 if "Аура" in symptoms else 0
    ])
    symptom_score["migraine"] = migraine_score
    
    # Аллергический ринит
    rhinitis_score = sum([
        2 if "Чихание" in symptoms else 0,
        2 if "Ринорея" in symptoms else 0,
        2 if "Заложенность носа" in symptoms else 0,
        2 if "Зуд в носу" in symptoms else 0,
        2 if "Слезотечение" in symptoms else 0,
        1 if "Сезонность" in symptoms else 0
    ])
    symptom_score["allergic_rhinitis"] = rhinitis_score
    
    # Находим наиболее вероятный диагноз
    sorted_diagnoses = sorted(symptom_score.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_diagnoses[0][0], sorted_diagnoses

# ОСНОВНОЙ ИНТЕРФЕЙС
def main():
    st.title("Медицинский справочник KazNMU [Камалов Жандос ОМ24-015]")
    st.markdown("**Комплексная система диагностики и рекомендаций по лечению**")
    
    # ВВОД ДАННЫХ
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Клиническая картина")
        
        symptoms = st.multiselect(
            "Симптомы пациента:",
            [
                "Лихорадка >38°C", "Озноб", "Кашель", "Кашель с мокротой", 
                "Одышка", "Боль в груди", "Боль в горле", "Налеты на миндалинах", 
                "Увеличение лимфоузлов", "Дизурия", "Учащенное мочеиспускание",
                "Боль в надлобковой области", "Тошнота", "Рвота", "Диарея",
                "Боль в животе", "Головная боль", "Пульсирующая головная боль",
                "Односторонняя локализация", "Тошнота/рвота", "Фоно/фотофобия",
                "Аура", "Чихание", "Ринорея", "Заложенность носа", "Зуд в носу",
                "Слезотечение", "Мышечные боли", "Слабость", "Внезапное начало",
                "Сезонность", "Субфебрильная температура", "Нарушение зрения"
            ]
        )
        
        temperature = st.slider("Температура тела (°C):", 35.0, 42.0, 37.0, 0.1)
        
    with col2:
        st.subheader("Лабораторные показатели")
        
        wbc = st.number_input("Лейкоциты (×10⁹/л):", min_value=1.0, max_value=50.0, value=6.0, step=0.1,
                             help="Норма: 4.0-9.0 ×10⁹/л")
        
        crp = st.number_input("СРБ (мг/л):", min_value=0.0, max_value=200.0, value=2.0, step=0.1,
                             help="Норма: <5 мг/л")
        
        lab_data = st.multiselect(
            "Другие результаты анализов:",
            [
                "Лейкоциты в моче", "Нитриты в моче", "Анализы в норме"
            ]
        )
        
        st.subheader("Артериальное давление")
        bp_col1, bp_col2 = st.columns(2)
        with bp_col1:
            bp_systolic = st.number_input("Систолическое (мм рт.ст.):", 80, 250, 120)
        with bp_col2:
            bp_diastolic = st.number_input("Диастолическое (мм рт.ст.):", 50, 150, 80)
    
    # ДИАГНОСТИКА
    if st.button("Провести диагностику", type="primary"):
        if not symptoms:
            st.warning("Пожалуйста, введите симптомы пациента")
            return
            
        with st.spinner("Провожу анализ симптомов..."):
            # Диагностика
            vital_signs = f"Температура: {temperature}°C, АД: {bp_systolic}/{bp_diastolic} мм рт.ст."
            main_diagnosis, all_diagnoses = medical_diagnosis_system(
                symptoms, lab_data, vital_signs, temperature, bp_systolic, bp_diastolic, wbc, crp
            )
            
            # РЕЗУЛЬТАТЫ
            st.markdown("---")
            st.subheader("Результаты диагностики")
            
            # Основной диагноз
            diagnosis_info = MEDICAL_KNOWLEDGE_BASE[main_diagnosis]
            diagnosis_name = main_diagnosis.replace('_', ' ').title()
            
            st.success(f"Основной диагноз: {diagnosis_name}")
            st.write(f"Баллы диагностики: {all_diagnoses[0][1]}/10")
            st.write(f"Источник рекомендаций: {diagnosis_info['source']}")
            
            # КРИТИЧЕСКИЕ СОСТОЯНИЯ
            if main_diagnosis == "hypertensive_crisis":
                st.error("КРИТИЧЕСКОЕ СОСТОЯНИЕ!")
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.write("НЕОБХОДИМО:")
                st.write("1. Немедленный вызов скорой помощи")
                st.write("2. Контроль АД каждые 15 минут")
                st.write("3. Покой, полусидячее положение")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # ЛЕЧЕНИЕ
            st.subheader("Рекомендации по лечению")
            
            treatments = diagnosis_info["treatments"]
            
            if "antibiotics" in treatments:
                st.markdown("**Антибактериальная терапия:**")
                for med in treatments["antibiotics"]:
                    st.write(f"- {med}")
            
            if "antivirals" in treatments:
                st.markdown("**Противовирусная терапия:**")
                for med in treatments["antivirals"]:
                    st.write(f"- {med}")
            
            if "antihistamines" in treatments:
                st.markdown("**Антигистаминные препараты:**")
                for med in treatments["antihistamines"]:
                    st.write(f"- {med}")
            
            if "rehydration" in treatments:
                st.markdown("**Регидратация:**")
                for med in treatments["rehydration"]:
                    st.write(f"- {med}")
            
            if "emergency" in treatments:
                st.markdown("**Неотложная помощь:**")
                for action in treatments["emergency"]:
                    st.write(f"- {action}")
            
            if "acute" in treatments:
                st.markdown("**Купирование острого приступа:**")
                for med in treatments["acute"]:
                    st.write(f"- {med}")
            
            st.markdown("**Симптоматическое лечение:**")
            if "symptomatic" in treatments:
                for med in treatments["symptomatic"]:
                    st.write(f"- {med}")
            
            st.markdown("**Вспомогательная терапия:**")
            if "supportive" in treatments:
                for action in treatments["supportive"]:
                    st.write(f"- {action}")
            
            if "diet" in treatments:
                st.markdown("**Диетические рекомендации:**")
                for item in treatments["diet"]:
                    st.write(f"- {item}")
            
            if "nasal" in treatments:
                st.markdown("**Назальная терапия:**")
                for med in treatments["nasal"]:
                    st.write(f"- {med}")
            
            if "avoidance" in treatments:
                st.markdown("**Элиминационные мероприятия:**")
                for action in treatments["avoidance"]:
                    st.write(f"- {action}")
            
            # НАПРАВЛЕНИЯ
            st.markdown("**Дальнейшие действия:**")
            st.info(diagnosis_info["referral"])
            
            # ДИФФЕРЕНЦИАЛЬНАЯ ДИАГНОСТИКА
            st.markdown("---")
            st.subheader("Дифференциальная диагностика")
            
            for i, (diagnosis, score) in enumerate(all_diagnoses[1:4], 1):
                diag_name = diagnosis.replace('_', ' ').title()
                st.write(f"{i}. {diag_name} ({score} баллов)")
    
    # ИНФОРМАЦИЯ О СИСТЕМЕ
    with st.sidebar:
        st.markdown("---")
        st.subheader("О системе")
        st.markdown("""
        **Диагностируемые состояния:**
        - Пневмония
        - Ангина
        - Инфекции МВП
        - Бронхит
        - Грипп
        - Гастроэнтерит
        - Гипертонический криз
        - Мигрень
        - Аллергический ринит
        
        **Основано на международных рекомендациях**
        """)
        
        st.markdown("---")
        st.subheader("Важно!")
        st.markdown("""
        Данная система предназначена для образовательных целей и не заменяет консультацию врача.
        
        При критических состояниях немедленно обращайтесь за медицинской помощью!
        """)

if __name__ == "__main__":
    main()
