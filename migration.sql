-- CreateTable
CREATE TABLE "registered_faces" (
    "id" UUID NOT NULL,
    "deleted_at" TIMESTAMPTZ(6),
    "updated_at" TIMESTAMPTZ(6),
    "created_at" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "label" VARCHAR(255) NOT NULL,
    "reference_id" VARCHAR(100) NOT NULL,

    CONSTRAINT "registered_faces_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "registered_faces_data" (
    "id" UUID NOT NULL,
    "registered_face_id" UUID NOT NULL,
    "deleted_at" TIMESTAMPTZ(6),
    "updated_at" TIMESTAMPTZ(6),
    "created_at" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "data" BYTEA NOT NULL,
    "pitch" REAL NOT NULL,
    "roll" REAL NOT NULL,
    "yaw" REAL NOT NULL,

    CONSTRAINT "registered_faces_data_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "trained_face" (
    "id" UUID NOT NULL,
    "registered_face_id" UUID NOT NULL,
    "data" JSON NOT NULL,
    "deleted_at" TIMESTAMPTZ(6),
    "updated_at" TIMESTAMPTZ(6),
    "created_at" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "trained_face_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "registered_faces_id_key" ON "registered_faces"("id");

-- CreateIndex
CREATE UNIQUE INDEX "registered_faces_data_id_key" ON "registered_faces_data"("id");

-- CreateIndex
CREATE UNIQUE INDEX "trained_face_id_key" ON "trained_face"("id");

-- AddForeignKey
ALTER TABLE "registered_faces_data" ADD CONSTRAINT "registered_faces_data_registered_face_id_fkey" FOREIGN KEY ("registered_face_id") REFERENCES "registered_faces"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "trained_face" ADD CONSTRAINT "trained_face_registered_face_id_fkey" FOREIGN KEY ("registered_face_id") REFERENCES "registered_faces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
