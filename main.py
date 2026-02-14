import torch  # бібліотека для роботи з тензорами та нейронними мережами
from diffusers import StableDiffusionPipeline  # імпорт класу для роботи з моделлю Stable Diffusion
import os  # бібліотека для роботи з операційною системою
import time  # бібліотека для роботи з часом
import traceback  # бібліотека для отримання інформації про помилки та їх трасування

MODEL_CACHE_DIR = "./models/stable-diffusion"  # директорія для збереження кешу моделі
MODEL_ID = "runwayml/stable-diffusion-v1-5"  # ідентифікатор моделі для завантаження


def download_model():
    print("Завантаження моделі Stable Diffusion...")
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)  # створення директорії для кешу, якщо вона не існує

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE_DIR,
            torch_dtype=torch.float16,  # використання 16-бітної точності для зменшення використання пам'яті
            safety_checker=None,  # відключення перевірки безпеки для прискорення завантаження
            use_safetensors=True,
            # використання формату safetensors для збереження моделі, що може бути швидше та безпечніше
            low_cpu_mem_usage=True,  # оптимізація використання пам'яті при завантаженні моделі
            variant="fp16"  # використання варіанту моделі з 16-бітною точністю
        )
        save_path = os.path.join(MODEL_CACHE_DIR, "saved_model")  # шлях для збереження моделі
        pipe.save_pretrained(save_path)  # збереження моделі у вказаному шляху
        print(f"Модель успішно завантажена та збережена в {save_path}")
        return True
    except Exception as e:
        print("Помилка при завантаженні моделі:", e)
        traceback.print_exc()  # виведення детальної інформації про помилку
        return False


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"  # визначення пристрою для завантаження моделі (GPU або CPU)
    print(f"Завантаження моделі на пристрій: {device}...")

    local = False
    if os.path.exists(MODEL_CACHE_DIR):
        local = True

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        cache_dir=MODEL_CACHE_DIR if local else None,
        use_safetensors=True,
        # використання кешу, якщо модель вже була завантажена раніше
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        # використання 16-бітної точності для GPU та 32-бітної для CPU
        safety_checker=None,  # відключення перевірки безпеки для прискор
        local_files_only=local,
        low_cpu_mem_usage=True,
        variant="fp16"
    ).to(device)  # завантаження моделі на вказаний пристрій

    if device == "cuda":
        pipe.enable_attention_slicing()  # оптимізація використання пам'яті на GPU шляхом розбиття уваги на частини

    return pipe


def generate_image(pipe, prompt, output_path=f"generated_image_{time.time()}.png", steps=100, guidance_scale=4.0):
    print("Генерація зображення за допомогою Stable Diffusion...")
    try:
        image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance_scale).images[
            0]  # генерація зображення на основі текстового запиту
        image.save(output_path)  # збереження згенерованого зображення у вказаному шляху
        print(f"Зображення успішно згенеровано та збережено в {output_path}")
        return image
    except Exception as e:
        print("Помилка при генерації зображення:", e)
        traceback.print_exc()  # виведення детальної інформації про помилку
        return None


def main():
    prompt = input("Введіть текстовий запит для генерації зображення: ")  # отримання текстового запиту від користувача
    try:
        pipe = load_model()  # завантаження моделі
        generate_image(pipe, prompt)  # генерація зображення на основі запиту
    except Exception as e:
        print("Помилка у головній функції:", e)
        traceback.print_exc()  # виведення детальної інформації про помилку


if __name__ == "__main__":
    # download_model()
    main()  # запуск головної функції
