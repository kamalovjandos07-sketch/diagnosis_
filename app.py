import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Настройки страницы
st.set_page_config(
    page_title="AI Medical Diagnostician",
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
    .guideline-ref { 
        background: #e8f5e8; 
        padding: 10px; 
        border-radius: 5px; 
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# 🏥 БАЗА КЛИНИЧЕСКИХ РЕКОМЕНДАЦИЙ И ДИАГНОСТИКИ
MEDICAL_KNOWLEDGE_BASE = {
    "community_acquired_pneumonia": {
        "diagnosis_criteria": ["Лихорадка >38°C", "Кашель", "Одышка", "Лейкоцитоз >10×10⁹/л", "Повышение СРБ >20 мг/л"],
        "required_criteria": 3,
        "antibiotic_choice": "Амоксициллин/клавуланат 875/125 мг 2 раза/сут × 7-10 дней",
        "source": "IDSA/ATS Guidelines 2019"
    },
    "streptococcal_pharyngitis": {
        "diagnosis_criteria": ["Боль в горле", "Лихорадка >38°C", "Налеты на миндалинах", "Увеличение шейных лимфоузлов", "Отсутствие кашля"],
        "required_criteria": 4, 
        "antibiotic_choice": "Феноксиметилпенициллин 500 мг 3 раза/сут × 10 дней",
        "source": "IDSA Pharyngitis Guidelines 2012"
    },
    "urinary_tract_infection": {
        "diagnosis_criteria": ["Дизурия", "Учащенное мочеиспускание", "Лихорадка >38°C", "Лейкоциты в моче", "Положительный нитритный тест"],
        "required_criteria": 2,
        "antibiotic_choice": "Цефтриаксон 1 г/сут в/м × 7 дней",
        "source": "IDSA UTI Guidelines 2022"
    },
    "acute_bronchitis": {
        "diagnosis_criteria": ["Кашель <3 недель", "Может быть продуктивным", "Отсутствие лихорадки >38°C", "Отсутствие одышки", "Нормальные показатели воспаления"],
        "required_criteria": 3,
        "antibiotic_choice": "Антибиотики НЕ ПОКАЗАНЫ - симптоматическая терапия",
        "source": "NICE Bronchitis Guidelines 2023"
    },
    "influenza": {
        "diagnosis_criteria": ["Внезапное начало", "Лихорадка", "Головная боль", "Мышечные боли", "Сезонность"],
        "required_criteria": 3,
        "antibiotic_choice": "Антибиотики НЕ эффективны - противовирусная терапия",
        "source": "WHO Influenza Guidelines 2023"
    }
}

# 🧠 БЕСПЛАТНЫЙ ИИ-ДИАГНОСТ (правила + логика)
def ai_medical_diagnostician(symptoms, lab_data, vital_signs, temperature, wbc, crp):
    """
    Бесплатный ИИ на основе медицинских правил и логики
    """
    
    # Анализируем симптомы
    symptom_score = {}
    
    # Пневмония
    pneumonia_score = sum([
        2 if "Лихорадка >38°C" in symptoms and temperature > 38 else 0,
        2 if "Кашель с мокротой" in symptoms else 1 if "Кашель" in symptoms else 0,
        2 if "Одышка" in symptoms else 0,
        2 if "Лейкоцитоз >10×10⁹/л" in lab_data and wbc > 10 else 0,
        2 if "Повышение СРБ >20 мг/л" in lab_data and crp > 20 else 0
    ])
    symptom_score["Пневмония"] = pneumonia_score
    
    # Ангина
    pharyngitis_score = sum([
        2 if "Боль в горле" in symptoms else 0,
        2 if "Налеты на миндалинах" in symptoms else 0,
        2 if "Лихорадка >38°C" in symptoms and temperature > 38 else 0,
        2 if "Увеличение лимфоузлов" in symptoms else 0,
        -2 if "Кашель" in symptoms else 1  # Кашель против стрептококковой этиологии
    ])
    symptom_score["Стрептококковая ангина"] = pharyngitis_score
    
    # ИМП
    uti_score = sum([
        3 if "Дизурия" in symptoms else 0,
        2 if "Учащенное мочеиспускание" in symptoms else 0,
        2 if "Лейкоциты в моче" in lab_data else 0,
        2 if "Лихорадка >38°C" in symptoms and temperature > 38 else 0
    ])
    symptom_score["Инфекция мочевых путей"] = uti_score
    
    # Бронхит
    bronchitis_score = sum([
        2 if "Кашель" in symptoms else 0,
        2 if "Кашель с мокротой" in symptoms else 0,
        -2 if "Лихорадка >38°C" in symptoms and temperature > 38 else 1,
        -2 if "Одышка" in symptoms else 1,
        -2 if "Лейкоцитоз >10×10⁹/л" in lab_data and wbc > 10 else 1
    ])
    symptom_score["Острый бронхит"] = bronchitis_score
    
    # Грипп
    influenza_score = sum([
        2 if "Лихорадка >38°C" in symptoms and temperature > 38 else 0,
        2 if "Головная боль" in symptoms else 0,
        2 if "Мышечные боли" in symptoms else 0,
        2 if "Внезапное начало" in symptoms else 0,
        1 if "Слабость" in symptoms else 0
    ])
    symptom_score["Грипп"] = influenza_score
    
    # Сортируем по вероятности
    sorted_diagnoses = sorted(symptom_score.items(), key=lambda x: x[1], reverse=True)
    
    # Формируем ответ
    result = "ВЕРОЯТНЫЙ ДИАГНОЗ: {}\n\n".format(sorted_diagnoses[0][0])
    result += "БАЛЛЫ ДИАГНОСТИКИ: {}/10\n\n".format(sorted_diagnoses[0][1])
    result += "ДИФФЕРЕНЦИАЛЬНАЯ ДИАГНОСТИКА:\n"
    
    for i, (diagnosis, score) in enumerate(sorted_diagnoses[1:4], 1):
        result += "{}. {} ({} баллов)\n".format(i, diagnosis, score)
    
    result += "\nОБОСНОВАНИЕ: Диагноз основан на анализе симптомов и данных обследования согласно клиническим рекомендациям."
    
    return result, sorted_diagnoses[0][0]

# 🔍 ПРОВЕРКА ПО КЛИНИЧЕСКИМ РЕКОМЕНДАЦИЯМ
def check_with_guidelines(diagnosis, symptoms, lab_data):
    """
    Проверяет диагноз по базе клинических рекомендаций
    """
    results = []
    
    diagnosis_key = diagnosis.lower().replace(" ", "_")
    
    for condition, guideline in MEDICAL_KNOWLEDGE_BASE.items():
        # Проверяем соответствие критериям
        matching_criteria = []
        for criterion in guideline["diagnosis_criteria"]:
            if any(symptom in criterion for symptom in symptoms) or any(lab in criterion for lab in lab_data):
                matching_criteria.append(criterion)
        
        if len(matching_criteria) >= guideline["required_criteria"]:
            results.append({
                "condition": condition,
                "matching_criteria": matching_criteria,
                "total_criteria": len(guideline["diagnosis_criteria"]),
                "guideline": guideline
            })
    
    return results

# 🎯 ОСНОВНОЙ ИНТЕРФЕЙС
def main():
    st.title("🩺 AI Medical Diagnostician")
    st.markdown("**Система диагностики на основе клинических рекомендаций и медицинской логики**")
    
    # 📝 ВВОД ДАННЫХ
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Клиническая картина")
        
        symptoms = st.multiselect(
            "Симптомы пациента:",
            [
                "Лихорадка >38°C", "Озноб", "Кашель", "Кашель с мокротой", 
                "Одышка", "Боль в горле", "Налеты на миндалинах", 
                "Дизурия", "Учащенное мочеиспускание", "Головная боль", 
                "Слабость", "Увеличение лимфоузлов", "Мышечные боли",
                "Внезапное начало"
            ]
        )
        
        temperature = st.slider("Температура тела (°C):", 35.0, 42.0, 37.0, 0.1)
        
    with col2:
        st.subheader("Данные обследования")
        
        lab_data = st.multiselect(
            "Результаты анализов:",
            [
                "Лейкоцитоз >10×10⁹/л", "Повышение СРБ >20 мг/л",
                "Лейкоциты в моче", "Нитриты в моче", "Анализы в норме"
            ]
        )
        
        wbc = st.number_input("Лейкоциты (×10⁹/л):", min_value=1.0, max_value=50.0, value=6.0)
        crp = st.number_input("СРБ (мг/л):", min_value=0.0, max_value=200.0, value=2.0)
    
    # 🔍 ДИАГНОСТИКА
    if st.button("🎯 Запустить диагностику", type="primary"):
        if not symptoms:
            st.warning("Пожалуйста, введите симптомы пациента")
            return
            
        with st.spinner("🩺 Анализирую симптомы по клиническим рекомендациям..."):
            # Формируем данные
            vital_signs = f"Температура: {temperature}°C"
            lab_info = f"Лейкоциты: {wbc}, СРБ: {crp}"
            
            # Получаем диагноз от ИИ
            ai_result, main_diagnosis = ai_medical_diagnostician(symptoms, lab_data, vital_signs, temperature, wbc, crp)
            
            # Проверяем по рекомендациям
            guideline_check = check_with_guidelines(main_diagnosis, symptoms, lab_data)
            
            # 📊 ВЫВОД РЕЗУЛЬТАТОВ
            st.markdown("---")
            st.subheader("🔍 Результаты диагностики")
            
            # Диагноз от ИИ
            st.markdown("### Заключение диагностической системы:")
            st.info(ai_result)
            
            # Проверка по рекомендациям
            st.markdown("### ✅ Проверка по клиническим рекомендациям:")
            
            if guideline_check:
                for i, result in enumerate(guideline_check[:3], 1):
                    with st.container():
                        st.markdown(f"#### {i}. {result['condition'].replace('_', ' ').title()}")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric(
                                "Соответствие критериям", 
                                f"{len(result['matching_criteria'])}/{result['total_criteria']}"
                            )
                        with col_b:
                            st.metric("Рекомендации", result['guideline']['source'])
                        
                        st.markdown(f"**Рекомендуемая терапия:** {result['guideline']['antibiotic_choice']}")
                        
                        with st.expander("📚 Критерии диагноза"):
                            for criterion in result['matching_criteria']:
                                st.markdown(f"✅ {criterion}")
                                
                        st.markdown("---")
            else:
                st.warning("❌ Случай не соответствует стандартным клиническим рекомендациям. Требуется консультация специалиста.")
    
    # 📚 ИНФОРМАЦИЯ О СИСТЕМЕ
    with st.sidebar:
        st.markdown("---")
        st.subheader("📖 О системе")
        st.markdown("""
        **Основано на рекомендациях:**
        - IDSA (Infectious Diseases Society of America)
        - NICE (National Institute for Health Care Excellence)  
        - WHO (Всемирная организация здравоохранения)
        
        **Бесплатная версия с медицинской логикой**
        """)
        
        st.markdown("---")
        st.subheader("🎓 Для студентов")
        st.markdown("""
        Эта система помогает:
        - Изучать диагностические критерии
        - Обучаться дифференциальной диагностике
        - Осваивать принципы доказательной медицины
        """)

if __name__ == "__main__":
    main()
