Cuda_convolution_image
======

Поддержка Microsoft .NET Framework 4.7.2 и Cuda 11.0

# О проекте
Cuda_convolution_image - Это windows form приложение написанное на языке c# для применения к изображению матрицы свёртки (размер матрицы от 3х3 до 9х9), где все вычисления производятся на видеокарте Nvidia. 
> Если видеокарта не имеет Cuda ядер (AMD видеокарты), то программа работать не будет.

Пример того, как выглядит применение свёртки:
![](https://zigorewslike.github.io/github_rep/cuda_conv/aply_conv.png)

# Открытие проекта

## Описание проектов
В одном решении присутствуют два проекта. Первый проект `cuda_convolution_UI` является основной программой (для открытия в IDE нужно установить Microsoft .NET Framework 4.7.2). Данному проекту для запуска требуется файл `cudaFanc64.dll`, который можно получить путём сборки второго проекта - `kernal_to_dll`. 

Таким образом порядок сборки проектов таков: 
1. kernal_to_dll
2. cuda_convolution_UI

## Открытие проекта kernal_to_dll
Для правильного запуска проекта (в IDE) нужно установить Cuda версию 11.0. 
Если на пк стоит другая версия Cuda, можно сменить версию в проекте. Для этого нужно открыть файл `kernel_to_dll.vcxproj` и отредактировать блок кода:
```xml
<ImportGroup Label="ExtensionTargets">
    <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 11.0.targets" />
</ImportGroup>
```
Нужно заменить `$(VCTargetsPath)\BuildCustomizations\CUDA 11.0.targets` на любую другую версию. Т.е. `$(VCTargetsPath)\BuildCustomizations\CUDA X.X.targets`, где X.X - версия cuda.

# Запуск проекта
Проект представляет собой оконное приложение, в котором нужно открыть изображение, составить матрицу свёртки или использовать пример, применить матрицу свёртки и по желанию сохранить картинку.



